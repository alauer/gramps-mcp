# gramps-mcp - AI-Powered Genealogy Research & Management
# Copyright (C) 2025 cabout.me
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
MCP server main entry point with HTTP transport.

This module provides the FastAPI application and MCP server setup with
all 23 genealogy tools for Gramps Web API integration.
"""

import asyncio
import logging
import os
import sys
from typing import Any, Dict, Optional

import anyio
from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp.types import Tool
from pydantic import BaseModel, Field

# Import all parameter models
from .models.parameters.citation_params import CitationData
from .models.parameters.event_params import EventSaveParams
from .models.parameters.family_params import FamilySaveParams
from .models.parameters.media_params import MediaSaveParams
from .models.parameters.note_params import NoteSaveParams
from .models.parameters.people_params import PersonData
from .models.parameters.place_params import PlaceSaveParams
from .models.parameters.repository_params import RepositoryData
from .models.parameters.simple_params import (
    SimpleFindParams,
    SimpleGetParams,
    SimpleSearchParams,
)
from .models.parameters.source_params import SourceSaveParams
from .models.parameters.transactions_params import TransactionHistoryParams

# Import all tool functions
from .tools import (
    create_citation_tool,
    create_event_tool,
    create_family_tool,
    create_media_tool,
    create_note_tool,
    create_person_tool,
    create_place_tool,
    create_repository_tool,
    create_source_tool,
    find_anything_tool,
    get_ancestors_tool,
    get_descendants_tool,
    get_recent_changes_tool,
    get_tree_info_tool,
)
from .tools.search_basic import find_type_tool
from .tools.search_details import get_type_tool


# Simple analysis models for tools that use direct dict access
class TreeInfoParams(BaseModel):
    include_statistics: bool = Field(True, description="Include statistics")


class DescendantsParams(BaseModel):
    gramps_id: str = Field(..., description="Person ID")
    max_generations: Optional[int] = Field(
        5,
        description=(
            "Max generations to retrieve (default: 5, use higher values "
            "carefully as they can overflow context)"
        ),
    )


class AncestorsParams(BaseModel):
    gramps_id: str = Field(..., description="Person ID")
    max_generations: Optional[int] = Field(
        5,
        description=(
            "Max generations to retrieve (default: 5, use higher values "
            "carefully as they can overflow context)"
        ),
    )


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Tool registry - single source of truth for all tools
TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Search & Retrieval Tools
    "find_type": {
        "description": (
            "Search any entity type using GQL - read gql://documentation "
            "resource first to understand syntax"
        ),
        "schema": SimpleFindParams,
        "handler": find_type_tool,
    },
    "find_anything": {
        "description": (
            "Text search across all record types - matches literal text "
            "within records, not logical combinations"
        ),
        "schema": SimpleSearchParams,
        "handler": find_anything_tool,
    },
    "get_type": {
        "description": "Get full details for person or family by handle or gramps_id",
        "schema": SimpleGetParams,
        "handler": get_type_tool,
    },
    # Data Management Tools
    "create_person": {
        "description": (
            "Create or update person information including family links "
            "and event associations"
        ),
        "schema": PersonData,
        "handler": create_person_tool,
    },
    "create_family": {
        "description": "Create or update family unit including member relationships",
        "schema": FamilySaveParams,
        "handler": create_family_tool,
    },
    "create_event": {
        "description": (
            "Create or update life event including person/place associations"
        ),
        "schema": EventSaveParams,
        "handler": create_event_tool,
    },
    "create_place": {
        "description": "Create or update geographic location",
        "schema": PlaceSaveParams,
        "handler": create_place_tool,
    },
    "create_source": {
        "description": "Create or update source document",
        "schema": SourceSaveParams,
        "handler": create_source_tool,
    },
    "create_citation": {
        "description": "Create or update citation including object associations",
        "schema": CitationData,
        "handler": create_citation_tool,
    },
    "create_note": {
        "description": "Create or update textual note including object associations",
        "schema": NoteSaveParams,
        "handler": create_note_tool,
    },
    "create_media": {
        "description": "Create or update media files including object associations",
        "schema": MediaSaveParams,
        "handler": create_media_tool,
    },
    "create_repository": {
        "description": "Create or update repository information",
        "schema": RepositoryData,
        "handler": create_repository_tool,
    },
    # Analysis Tools
    "tree_stats": {
        "description": (
            "Get information about a specific tree including statistics "
            "(counts of people, families, events, etc.)"
        ),
        "schema": TreeInfoParams,
        "handler": get_tree_info_tool,
    },
    "get_descendants": {
        "description": (
            "Find all descendants of a person - WARNING: Very token-heavy "
            "operation, minimize generations (default: 5)"
        ),
        "schema": DescendantsParams,
        "handler": get_descendants_tool,
    },
    "get_ancestors": {
        "description": (
            "Find all ancestors of a person - WARNING: Very token-heavy "
            "operation, minimize generations (default: 5)"
        ),
        "schema": AncestorsParams,
        "handler": get_ancestors_tool,
    },
    "recent_changes": {
        "description": "Get recent changes/modifications to the family tree",
        "schema": TransactionHistoryParams,
        "handler": get_recent_changes_tool,
    },
}


# Create FastMCP app with stateless HTTP (no SSE)
# Reason: json_response=True returns JSON instead of SSE for POST responses
# which is friendlier for non-streaming MCP clients and tooling.
app = FastMCP("gramps", stateless_http=True, json_response=True)


# ============================================================================
# Dynamic FastMCP Tool Registration
# ============================================================================


class _ToolCallable:
    """Wrapper that exposes a clean ``(arguments)`` signature to FastMCP.

    FastMCP introspects ``inspect.signature`` of registered tool callables to
    build the JSON schema. If a Python closure captures the real handler as a
    default-valued parameter (e.g. ``def f(arguments, handler=real): ...``),
    Pydantic's schema generator emits a non-serializable default warning and
    leaks an internal ``handler`` field of type ``"string"`` into the public
    tool schema. Using a callable instance with a single-parameter
    ``__call__`` keeps the signature clean and matches the MCP spec, where
    the tool input is the schema object directly.

    The concrete schema is bound to ``__call__.__annotations__`` so FastMCP
    can pick it up via ``inspect.signature(..., eval_str=True)`` and build a
    rich Pydantic model (preserving nested types and field descriptions).
    The annotation is set on the unbound function object so it survives the
    bound-method wrapping that Python applies at attribute lookup.

    Attributes:
        __name__: Used by FastMCP as the tool's registered name.
        __doc__: Used by FastMCP as the tool's description.
    """

    def __init__(self, name: str, description: str, schema, handler):
        self.__name__ = name
        self.__doc__ = description
        self._handler = handler
        # Bind the Pydantic schema as the ``arguments`` parameter type. We
        # set the annotation on the underlying function (``__call__``) rather
        # than the bound method so it persists for ``inspect.signature``.
        self.__call__.__annotations__["arguments"] = schema

    async def __call__(self, arguments):
        """Dispatch the validated Pydantic model to the real tool handler."""
        return await self._handler(arguments.model_dump())


def register_tools():
    """Register all tools from the registry with FastMCP.

    Each registered tool accepts a single ``arguments`` field whose schema is
    the tool's Pydantic parameter model. The real handler is captured in a
    callable wrapper so FastMCP's introspection only sees the public
    ``(arguments)`` signature.
    """
    for tool_name, tool_config in TOOL_REGISTRY.items():
        description = tool_config["description"]
        schema = tool_config["schema"]
        handler_func = tool_config["handler"]
        wrapper = _ToolCallable(tool_name, description, schema, handler_func)

        # Register with FastMCP using the description as the tool's
        # user-facing documentation.
        app.tool(description=description, name=tool_name)(wrapper)


register_tools()


# ============================================================================
# HTTP Compatibility Middleware
# ============================================================================


class HttpCompatibilityMiddleware:
    """ASGI middleware that fills HTTP compatibility gaps for MCP clients.

    The MCP Streamable HTTP transport only handles ``GET``, ``POST``, and
    ``DELETE`` on the ``/mcp`` endpoint. Two real-world client flows are
    rejected with HTTP 405 by the transport:

    1. **CORS preflight (OPTIONS)** - Browser-based clients (Open WebUI,
       ``mcpo``, in-browser MCP inspectors) must send a CORS preflight
       request before their first POST. Returning 405 here stops the
       browser from completing the handshake.
    2. **HEAD probes** - Container orchestrators and HTTP health probes
       use HEAD to verify the server is up. The MCP transport returns 405
       for HEAD, which orchestrators interpret as ``"endpoint not
       available"``.

    This middleware short-circuits those two methods on the MCP path with
    the headers each caller expects, then delegates everything else to
    the underlying MCP application.

    Attributes:
        app: The wrapped ASGI application (the MCP streamable HTTP app).
        mcp_path: The configured MCP endpoint path (e.g. ``"/mcp"``).
    """

    def __init__(self, app, mcp_path: str):
        self.app = app
        self.mcp_path = mcp_path

    async def __call__(self, scope, receive, send):
        # Only intercept on the configured HTTP scope and MCP path.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path") != self.mcp_path:
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "").upper()

        if method == "OPTIONS":
            await self._handle_options(scope, send)
            return

        if method == "HEAD":
            await self._handle_head(scope, send)
            return

        # Pass through everything else (GET, POST, DELETE).
        await self.app(scope, receive, send)

    async def _handle_options(self, scope, send):
        """Respond to a CORS preflight with the MCP-required headers.

        The browser sends an ``Access-Control-Request-Method`` header
        advertising the method it intends to use. We mirror back
        everything the transport would allow plus the common MCP
        request/response headers used by clients.
        """
        request_headers = _collect_request_headers(scope)
        requested_method = request_headers.get(
            "access-control-request-method", "POST"
        )
        requested_headers = request_headers.get(
            "access-control-request-headers", ""
        )

        # Mirror the caller's origin if it sent one, otherwise fall back
        # to a wildcard. Browsers reject ``*`` when credentials are
        # present, so reflecting the origin is the safe default.
        origin = request_headers.get("origin", "*")
        allow_origin = origin if origin != "*" else "*"

        headers = [
            (b"access-control-allow-origin", allow_origin.encode("latin-1")),
            (
                b"access-control-allow-methods",
                b"GET, POST, DELETE, OPTIONS, HEAD",
            ),
            (b"access-control-allow-headers", requested_headers.encode("latin-1") or b"Content-Type, Accept, mcp-session-id, mcp-protocol-version, last-event-id"),
            (b"access-control-max-age", b"600"),
            (b"access-control-expose-headers", b"mcp-session-id, mcp-protocol-version"),
            (b"vary", b"Origin, Access-Control-Request-Method, Access-Control-Request-Headers"),
            (b"content-length", b"0"),
        ]

        # If the request asked for credentials, allow them. Browsers will
        # also accept the reflected origin in that case.
        if request_headers.get("access-control-request-headers"):
            headers.append((b"access-control-allow-credentials", b"true"))

        await send({"type": "http.response.start", "status": 204, "headers": headers})
        await send({"type": "http.response.body", "body": b""})

    async def _handle_head(self, scope, send):
        """Reply to a HEAD probe with a minimal 200 response.

        Health-check probes only need to know the endpoint is reachable
        and the right CORS headers are present. We don't try to mirror
        the GET response body (which would require consuming the MCP
        streamable HTTP SSE flow) because probes do not inspect it.
        """
        request_headers = _collect_request_headers(scope)
        origin = request_headers.get("origin")
        headers = [
            (b"content-type", b"text/event-stream"),
            (b"cache-control", b"no-cache, no-transform"),
        ]
        if origin:
            headers.append(
                (b"access-control-allow-origin", origin.encode("latin-1"))
            )
            headers.append((b"vary", b"Origin"))

        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": b""})


def _collect_request_headers(scope) -> Dict[str, str]:
    """Return a case-insensitive dict of HTTP headers from an ASGI scope."""
    raw = scope.get("headers") or []
    headers: Dict[str, str] = {}
    for name, value in raw:
        try:
            key = name.decode("latin-1").lower()
        except Exception:
            continue
        try:
            val = value.decode("latin-1")
        except Exception:
            val = ""
        headers[key] = val
    return headers


# ============================================================================
# Resource Management
# ============================================================================


def load_resource(filename: str) -> str:
    """Load content from resources folder with error handling."""
    try:
        # Get the path to the resources directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        resource_path = os.path.join(current_dir, "resources", filename)

        with open(resource_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Resource file '{filename}' not found."
    except Exception as e:
        return f"Error loading resource '{filename}': {str(e)}"


@app.resource("gql://documentation")
def get_gql_documentation() -> str:
    """
    Complete GQL documentation, syntax, examples, and property
    reference for Gramps queries.
    """
    return load_resource("gql-documentation.md")


@app.resource("gramps://usage-guide")
def get_usage_guide() -> str:
    """
    IMPORTANT: Read this first before using ANY creation tools -
    explains proper genealogy workflow and tool usage order.
    """
    return load_resource("gramps-usage-guide.md")


# Add custom routes to the FastMCP app
@app.custom_route("/", ["GET"])
async def root(request):
    """Root endpoint with server information."""
    from starlette.responses import JSONResponse

    return JSONResponse(
        {
            "service": "Gramps MCP Server",
            "version": "1.0.0",
            "description": "MCP server for Gramps Web API genealogy operations",
            "mcp_endpoint": "/mcp",
            "tools_count": 16,
        }
    )


@app.custom_route("/health", ["GET"])
async def health_check(request):
    """Health check endpoint."""
    from starlette.responses import JSONResponse

    return JSONResponse(
        {"status": "healthy", "service": "Gramps MCP Server", "tools": 16}
    )


async def run_stdio_server():
    """Run the MCP server with stdio transport."""
    # Create a standard MCP server for stdio transport
    server = Server("gramps")

    @server.list_tools()
    async def handle_list_tools():
        """List all available tools."""
        return [
            Tool(
                name=tool_name,
                description=tool_config["description"],
                inputSchema=tool_config["schema"].model_json_schema(),
            )
            for tool_name, tool_config in TOOL_REGISTRY.items()
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict):
        """Handle tool calls."""
        if name in TOOL_REGISTRY:
            return await TOOL_REGISTRY[name]["handler"](arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    # Run the server with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    # Determine transport type from command line arguments or environment
    transport_type = sys.argv[1] if len(sys.argv) > 1 else "streamable-http"

    if transport_type == "stdio":
        # Run with stdio transport for CLI usage
        asyncio.run(run_stdio_server())
    else:
        # Run the FastMCP server with streamable HTTP transport
        # Configure server settings
        app.settings.host = "0.0.0.0"  # Listen on all interfaces for Docker
        app.settings.port = 8000

        # Build the Starlette app, then add an HTTP compatibility middleware.
        # Reason: the underlying Streamable HTTP transport only accepts
        # GET, POST, and DELETE on the ``/mcp`` endpoint and returns 405
        # for everything else. This breaks two real-world client flows:
        #   1. Browser-based MCP clients (Open WebUI, mcpo, etc.) trigger
        #      an OPTIONS preflight which is rejected with 405.
        #   2. Health-check / monitoring probes use HEAD, which is also
        #      rejected with 405.
        # The middleware below short-circuits those two methods with
        # proper headers so the rest of the protocol still works.
        import uvicorn

        starlette_app = app.streamable_http_app()
        starlette_app.add_middleware(
            HttpCompatibilityMiddleware,
            mcp_path=app.settings.streamable_http_path,
        )
        config = uvicorn.Config(
            starlette_app,
            host=app.settings.host,
            port=app.settings.port,
            log_level=app.settings.log_level.lower(),
        )
        server = uvicorn.Server(config)
        anyio.run(server.serve)
