import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8081/v1",
    api_key="not-needed",          # llama-server doesn't check it
)

stream = client.chat.completions.create(
    model="Gemma4",
    messages=[{"role": "user", "content": "Tell me an interesting fact about Brazil. Keep it in one paragraph."}],
    stream=True,
    stream_options={"include_usage": True},   # usage arrives in the final chunk
)

for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
