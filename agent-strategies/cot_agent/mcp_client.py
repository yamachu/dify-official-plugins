"""
MCP (Model Context Protocol) HTTP client for remote tool integration.

Supports:
- HTTP/HTTPS communication with MCP servers
- JSON-RPC 2.0 message format
- SSE (Server-Sent Events) for streaming tool results
- Streamable HTTP with newline-delimited JSON
- Multiple MCP servers with tool merging
"""

import json
import logging
from typing import Any, Optional
from dataclasses import dataclass, field
from collections.abc import Generator
import httpx
from httpx_sse import EventSource, SSEError

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str
    url: str  # HTTP(S) URL to MCP server
    transport: str = "sse"  # Transport type: "sse", "streamable_http"
    auth_token: Optional[str] = None  # Optional bearer token
    headers: dict[str, str] = field(default_factory=dict)  # Custom headers
    session_id: Optional[str] = None  # MCP session ID


@dataclass
class MCPToolInfo:
    """Information about a tool from an MCP server."""
    server_name: str
    tool_name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    prefixed_name: str = ""

    def __post_init__(self):
        if not self.prefixed_name:
            self.prefixed_name = f"{self.server_name}:{self.tool_name}"


class MCPConfigError(Exception):
    """Error in MCP configuration or operation."""
    pass


class MCPClient:
    """HTTP-based client for managing MCP server connections and tool invocation."""

    def __init__(self, mcp_json_str: Optional[str] = None):
        """
        Initialize MCP client from JSON configuration.

        Args:
            mcp_json_str: JSON string containing mcpServers configuration.
                        Expected format:
                        {
                            "server-name": {
                                "url": "http://localhost:8000",
                                "transport": "sse",  # Optional: "sse" (default), "streamable_http"
                                "auth_token": "optional-bearer-token"
                            },
                            "another-server": {
                                "url": "https://mcp.example.com",
                                "transport": "streamable_http"
        """
        self.servers: dict[str, MCPServerConfig] = {}
        self.tools: dict[str, MCPToolInfo] = {}  # Prefixed tool name -> tool info
        self.tool_name_map: dict[str, tuple[str, str]] = {}  # Prefixed name -> (server_name, tool_name)
        self.http_client = httpx.Client(timeout=30)
        self._request_id = 0  # JSON-RPC request ID counter

        if not mcp_json_str or not mcp_json_str.strip():
            # MCP disabled
            self.enabled = False
            return

        self.enabled = True

        try:
            config = json.loads(mcp_json_str)
        except json.JSONDecodeError as e:
            raise MCPConfigError(f"Invalid MCP JSON configuration: {e}") from e

        if not isinstance(config, dict):
            raise MCPConfigError(
                f"MCP configuration must be a JSON object, got {type(config).__name__}"
            )

        for server_name, server_config in config.items():
            if not isinstance(server_config, dict):
                raise MCPConfigError(
                    f"Server '{server_name}' config must be an object, got {type(server_config).__name__}"
                )

            url = server_config.get("url")
            if not url:
                raise MCPConfigError(
                    f"Server '{server_name}' missing required 'url' field"
                )

            if not isinstance(url, str):
                raise MCPConfigError(
                    f"Server '{server_name}' 'url' must be a string, got {type(url).__name__}"
                )

            transport = server_config.get("transport", "sse")
            if transport not in ("sse", "streamable_http"):
                raise MCPConfigError(
                    f"Server '{server_name}' 'transport' must be one of 'sse' or 'streamable_http', got '{transport}'"
                )

            auth_token = server_config.get("auth_token")
            headers = server_config.get("headers", {})

            if not isinstance(headers, dict):
                raise MCPConfigError(
                    f"Server '{server_name}' 'headers' must be an object, got {type(headers).__name__}"
                )

            self.servers[server_name] = MCPServerConfig(
                name=server_name,
                url=url,
                transport=transport,
                auth_token=auth_token,
                headers=headers,
            )

    def discover_tools(self) -> dict[str, MCPToolInfo]:
        """
        Discover available tools from all configured MCP servers.

        Returns:
            Dictionary mapping prefixed tool names to MCPToolInfo.

        Raises:
            MCPConfigError: If tool discovery fails.
        """
        if not self.enabled or not self.servers:
            return {}

        if self.tools:
            # Already discovered
            return self.tools

        for server_name, server_config in self.servers.items():
            try:
                tools_list = self._fetch_tools_from_server(server_name, server_config)
                for tool_info in tools_list:
                    prefixed_name = f"{server_name}:{tool_info.tool_name}"
                    self.tools[prefixed_name] = MCPToolInfo(
                        server_name=server_name,
                        tool_name=tool_info.tool_name,
                        description=tool_info.description,
                        input_schema=tool_info.input_schema,
                        prefixed_name=prefixed_name,
                    )
                    self.tool_name_map[prefixed_name] = (server_name, tool_info.tool_name)
            except Exception as e:
                logger.error(f"Failed to discover tools from server '{server_name}': {e}")
                raise MCPConfigError(f"Tool discovery failed for server '{server_name}': {e}") from e

        return self.tools

    def _get_next_request_id(self) -> int:
        """Get the next JSON-RPC request ID."""
        self._request_id += 1
        return self._request_id

    def _parse_json_response(self, response: httpx.Response, server_name: str) -> dict:
        """
        Parse JSON response from MCP server (handles both plain JSON and SSE format).
        
        Args:
            response: HTTP response object.
            server_name: Name of the server (for error messages).
            
        Returns:
            Parsed JSON data.
            
        Raises:
            MCPConfigError: If response cannot be parsed.
        """
        try:
            response_text = response.text
            
            # Check if response is in SSE format
            if response_text.startswith("event:") or response_text.startswith("data:"):
                logger.debug(f"Response from '{server_name}' is in SSE format")
                # Parse SSE format: extract JSON from "data: " lines
                for line in response_text.split('\n'):
                    if line.startswith('data: '):
                        json_str = line[6:].strip()  # Remove "data: " prefix
                        if json_str:
                            return json.loads(json_str)
                
                raise MCPConfigError(f"No valid data line found in SSE response from '{server_name}'")
            else:
                # Plain JSON response
                return response.json()
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response from '{server_name}' as JSON")
            logger.error(f"Response body: {response.text[:500]}")
            raise MCPConfigError(
                f"Invalid JSON response from '{server_name}': {e}. "
                f"Response body: {response.text[:200]}"
            ) from e

    def _initialize_server(self, server_name: str, server_config: MCPServerConfig) -> None:
        """
        Initialize MCP server connection.
        
        Args:
            server_name: Name of the server.
            server_config: Server configuration.
        """
        try:
            headers = self._build_headers(server_config)
            
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "dify-mcp-client",
                        "version": "1.0.0"
                    }
                },
                "id": self._get_next_request_id(),
            }
            
            logger.debug(f"Initializing MCP server '{server_name}': {init_request}")
            logger.debug(f"Initialize headers: {headers}")
            
            response = self.http_client.post(
                server_config.url.rstrip('/'),
                json=init_request,
                headers=headers,
            )
            
            logger.debug(f"Initialize response status: {response.status_code}")
            logger.debug(f"Initialize response headers: {dict(response.headers)}")
            logger.debug(f"Initialize response body (raw): {response.text[:500]}")  # First 500 chars
            
            response.raise_for_status()
            
            # Store session ID if present
            if "mcp-session-id" in response.headers:
                server_config.session_id = response.headers["mcp-session-id"]
                logger.debug(f"Received session ID for '{server_name}': {server_config.session_id}")
            
            # Parse JSON response (may be SSE format or plain JSON)
            data = self._parse_json_response(response, server_name)
            logger.debug(f"Initialize response from '{server_name}': {data}")
            
            if "error" in data:
                error = data["error"]
                raise MCPConfigError(
                    f"Initialize error: {error.get('message', 'Unknown error')} "
                    f"(code: {error.get('code', 'N/A')})"
                )
            
            # Send initialized notification
            notify_request = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            
            notify_headers = self._build_headers(server_config)
            notify_response = self.http_client.post(
                server_config.url.rstrip('/'),
                json=notify_request,
                headers=notify_headers,
            )
            notify_response.raise_for_status()
            
            # Log notification response (notifications may not return data)
            if notify_response.text:
                logger.debug(f"Notification response: {notify_response.text[:200]}")
            
            logger.info(f"Successfully initialized MCP server '{server_name}'")
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error initializing server '{server_name}': {e}")
            if hasattr(e, 'response') and e.response:
                try:
                    error_body = e.response.text
                    logger.error(f"Error response body: {error_body}")
                except Exception:
                    pass
            raise MCPConfigError(f"Failed to initialize server '{server_name}': {e}") from e
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Invalid response from '{server_name}': {e}")
            raise MCPConfigError(f"Invalid initialize response from '{server_name}': {e}") from e

    def _fetch_tools_from_server(
        self, server_name: str, server_config: MCPServerConfig
    ) -> list[MCPToolInfo]:
        """
        Fetch tools from a specific MCP server via JSON-RPC.

        Args:
            server_name: Name of the server.
            server_config: Server configuration.

        Returns:
            List of tools available from the server.
        """
        try:
            # Initialize server if not already done (establishes session)
            if not server_config.session_id:
                self._initialize_server(server_name, server_config)
            
            headers = self._build_headers(server_config)
            # JSON-RPC request for tools/list (with empty params as per MCP spec)
            jsonrpc_request = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
                "id": self._get_next_request_id(),
            }
            
            logger.debug(f"Sending JSON-RPC request to '{server_name}': {jsonrpc_request}")
            
            response = self.http_client.post(
                server_config.url.rstrip('/'),
                json=jsonrpc_request,
                headers=headers,
                follow_redirects=True,
            )
            response.raise_for_status()

            data = self._parse_json_response(response, server_name)
            logger.debug(f"Received JSON-RPC response from '{server_name}': {data}")
            
            # Handle JSON-RPC response format
            if "error" in data:
                error = data["error"]
                raise MCPConfigError(
                    f"JSON-RPC error: {error.get('message', 'Unknown error')} "
                    f"(code: {error.get('code', 'N/A')})"
                )
            
            result_data = data.get("result", {})
            tools = result_data.get("tools", [])

            result = []
            for tool_dict in tools:
                tool_info = MCPToolInfo(
                    server_name=server_name,
                    tool_name=tool_dict.get("name", ""),
                    description=tool_dict.get("description", ""),
                    input_schema=tool_dict.get("inputSchema", tool_dict.get("input_schema", {})),
                )
                result.append(tool_info)

            logger.info(
                f"Discovered {len(result)} tools from MCP server '{server_name}' "
                f"(URL: {server_config.url})"
            )
            return result
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching tools from '{server_name}': {e}")
            if hasattr(e, 'response') and e.response:
                try:
                    error_body = e.response.text
                    logger.error(f"Error response body: {error_body}")
                except Exception:
                    pass
            raise
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Invalid response format from '{server_name}': {e}")
            raise

    def invoke_tool(
        self, prefixed_tool_name: str, parameters: dict[str, Any]
    ) -> Generator[dict[str, Any], None, None]:
        """
        Invoke a tool on an MCP server via HTTP and stream results via SSE.

        Args:
            prefixed_tool_name: Tool name prefixed with server name (e.g., "server:tool").
            parameters: Tool parameters.

        Yields:
            Chunks of tool output from SSE stream.

        Raises:
            MCPConfigError: If tool not found or invocation fails.
        """
        if not self.enabled:
            raise MCPConfigError("MCP is not enabled")

        if prefixed_tool_name not in self.tool_name_map:
            raise MCPConfigError(f"Tool '{prefixed_tool_name}' not found in MCP servers")

        server_name, tool_name = self.tool_name_map[prefixed_tool_name]
        server_config = self.servers[server_name]

        try:
            yield from self._invoke_on_server(server_name, server_config, tool_name, parameters)
        except Exception as e:
            logger.error(f"Failed to invoke tool '{prefixed_tool_name}' on server '{server_name}': {e}")
            raise MCPConfigError(
                f"Tool invocation failed for '{prefixed_tool_name}': {e}"
            ) from e

    def _invoke_on_server(
        self,
        server_name: str,
        server_config: MCPServerConfig,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> Generator[dict[str, Any], None, None]:
        """
        Invoke a tool on a specific server via HTTP.

        Args:
            server_name: Name of the server.
            server_config: Server configuration.
            tool_name: Name of the tool.
            parameters: Tool parameters.

        Yields:
            Tool output chunks based on transport type.
        """
        if server_config.transport == "sse":
            yield from self._invoke_sse(server_name, server_config, tool_name, parameters)
        elif server_config.transport == "streamable_http":
            yield from self._invoke_streamable_http(server_name, server_config, tool_name, parameters)
        else:
            raise MCPConfigError(f"Unsupported transport type: {server_config.transport}")

    def _invoke_sse(
        self,
        server_name: str,
        server_config: MCPServerConfig,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> Generator[dict[str, Any], None, None]:
        """
        Invoke a tool using SSE (Server-Sent Events) streaming with JSON-RPC.
        """
        headers = self._build_headers(server_config)
        headers["Accept"] = "application/json, text/event-stream"

        # JSON-RPC request for tools/call
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters,
            },
            "id": self._get_next_request_id(),
        }

        try:
            logger.debug(f"Invoking tool '{tool_name}' via JSON-RPC SSE: {jsonrpc_request}")
            with httpx.stream(
                "POST",
                server_config.url.rstrip('/'),
                json=jsonrpc_request,
                headers=headers,
                timeout=300,  # Longer timeout for streaming
                follow_redirects=True,
            ) as response:
                response.raise_for_status()
                yield from self._parse_sse_stream(response)
        except httpx.HTTPError as e:
            logger.error(f"HTTP error invoking tool '{tool_name}' on '{server_name}': {e}")
            raise

    def _invoke_streamable_http(
        self,
        server_name: str,
        server_config: MCPServerConfig,
        tool_name: str,
        parameters: dict[str, Any],
    ) -> Generator[dict[str, Any], None, None]:
        """
        Invoke a tool using streamable HTTP with JSON-RPC (chunked transfer encoding with newline-delimited JSON).
        """
        headers = self._build_headers(server_config)
        headers["Accept"] = "application/json, application/x-ndjson"

        # JSON-RPC request for tools/call
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": parameters,
            },
            "id": self._get_next_request_id(),
        }

        try:
            logger.debug(f"Invoking tool '{tool_name}' via JSON-RPC streamable HTTP: {jsonrpc_request}")
            with httpx.stream(
                "POST",
                server_config.url.rstrip('/'),
                json=jsonrpc_request,
                headers=headers,
                timeout=300,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError:
                            yield {"text": line}
        except httpx.HTTPError as e:
            logger.error(f"HTTP error invoking tool '{tool_name}' on '{server_name}': {e}")
            raise

    def _parse_sse_stream(self, response: httpx.Response) -> Generator[dict[str, Any], None, None]:
        """
        Parse SSE stream from MCP server response.

        Args:
            response: HTTP response with SSE stream.

        Yields:
            Parsed SSE events as dictionaries.
        """
        try:
            event_source = EventSource(response)
            for event in event_source.iter_sse():
                if event.data:
                    try:
                        data = json.loads(event.data)
                        yield data
                    except json.JSONDecodeError:
                        # If not JSON, yield as text
                        yield {"text": event.data}
        except SSEError as e:
            logger.error(f"SSE error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing SSE stream: {e}")
            raise

    def _build_headers(self, server_config: MCPServerConfig) -> dict[str, str]:
        """
        Build HTTP headers for MCP server request.

        Args:
            server_config: Server configuration.

        Returns:
            Dictionary of headers.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **server_config.headers,
        }

        if server_config.auth_token:
            headers["Authorization"] = f"Bearer {server_config.auth_token}"
        
        # Include session ID if established
        if server_config.session_id:
            headers["Mcp-Session-Id"] = server_config.session_id

        return headers

    def close(self) -> None:
        """Close HTTP client and cleanup resources."""
        try:
            self.http_client.close()
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def parse_mcp_config(mcp_json_str: Optional[str]) -> MCPClient:
    """
    Parse MCP configuration from JSON string and return client.

    Args:
        mcp_json_str: JSON string with mcpServers configuration.

    Returns:
        MCPClient instance.

    Raises:
        MCPConfigError: If configuration is invalid.
    """
    return MCPClient(mcp_json_str)

