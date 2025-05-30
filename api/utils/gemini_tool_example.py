from google import genai
from google.genai import types

from api.utils.settings import get_setting

# Based on https://cloud.google.com/vertex-ai/generative-ai/docs/grounding/grounding-with-google-maps#json
config = types.GenerateContentConfig(
    tools=[
        types.Tool(
            google_maps=types.GoogleMaps(
                auth_config=types.AuthConfig(
                    api_key_config=types.ApiKeyConfig(
                        api_key_string=get_setting("GOOGLE_MAPS_API_KEY")
                    )
                )
            )
        )
    ],
    tool_config=types.ToolConfig(
        retrieval_config=types.RetrievalConfig(
            lat_lng=types.LatLng(latitude=40.7092777, longitude=-74.0178055)
        )
    ),
)

# Configure the client
client = genai.Client(api_key=get_setting("GEMINI_API_KEY"))

# Define user prompt
contents = [
    types.Content(
        role="user", parts=[types.Part(text="List 3 italian restaurants near me")]
    )
]

# Send request with function declarations
response = client.models.generate_content(
    model="gemini-2.0-flash", config=config, contents=contents
)

print(response.candidates[0].content.parts[0].function_call)
print("IF ALL ELSE FAILS HERE IS FULL RESPONSE")
print(response)
