from mcp import StdioServerParameters

from api.utils.settings import get_setting


def google_maps():
    # Inspired by https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#model_context_protocol_mcp
    # Create server parameters for stdio connection
    # This is the API: https://github.com/modelcontextprotocol/servers-archived/tree/main/src/google-maps
    return StdioServerParameters(
        command="npx",  # Executable
        # https://github.com/modelcontextprotocol/servers-archived
        args=["-y", "@modelcontextprotocol/server-google-maps"],  # MCP Server
        env={
            "GOOGLE_MAPS_API_KEY": get_setting("GOOGLE_MAPS_API_KEY"),
        },
    )
