import asyncio

from google import genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from api.utils.settings import get_setting

client = genai.Client(api_key=get_setting("GEMINI_API_KEY"))

# Inspired by https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#model_context_protocol_mcp
# Create server parameters for stdio connection
# This is the API: https://github.com/modelcontextprotocol/servers-archived/tree/main/src/google-maps
server_params = StdioServerParameters(
    command="npx",  # Executable
    # https://github.com/modelcontextprotocol/servers-archived
    args=["-y", "@modelcontextprotocol/server-google-maps"],  # MCP Server
    env={
        "GOOGLE_MAPS_API_KEY": get_setting("GOOGLE_MAPS_API_KEY"),
    },
)


async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Prompt to get the weather for the current day in London.
            prompt = f"Find 3 convenience stores near me. I am currently at 380 Rector Place 8P, New York, NY 10280"
            # prompt = f"Find 3 italian restaurants in NYC"
            # Initialize the connection between client and server
            await session.initialize()
            # Send request to the model with MCP function declarations
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],
                ),
            )
            print(response)


# Start the asyncio event loop and run the main function
asyncio.run(run())
