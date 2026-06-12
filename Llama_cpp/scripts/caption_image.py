import base64
import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8081/v1",
    api_key="not-needed",          # llama-server doesn't check it
)

IMG_PATH = "/home/mjrovai/Pictures/box_3_wheel_4.jpg"

def to_data_uri(path, mime="image/jpeg"):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

resp = client.chat.completions.create(
    model="Gemma4",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": to_data_uri(IMG_PATH)}},
        ],
    }],
)
print(resp.choices[0].message.content)
