"""Entry point for running the xraylarch MCP server."""

from xraylarch_mcp.server import mcp


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
