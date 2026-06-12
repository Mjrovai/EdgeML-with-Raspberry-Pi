import base64
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8081/v1",
    api_key="not-needed",          # llama-server doesn't check it
)

SOUND_PATH = "christmas_space_30s.mp3"

def b64_audio(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="gemma-4",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe this clip."},
            {"type": "input_audio",
             "input_audio": {"data": b64_audio(SOUND_PATH), "format": "mp3"}},
        ],
    }],
)
print(resp.choices[0].message.content)
