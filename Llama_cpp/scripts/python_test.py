import time
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8081/v1",
    api_key="not-needed",          # llama-server doesn't check it
)

t0 = time.perf_counter()
resp = client.chat.completions.create(
    model="Gemma4",               # ignored by llama-server, kept for readability
    messages=[{"role": "user", "content": "What is the capital of Brazil?"}],
)
dt = time.perf_counter() - t0

print(resp.choices[0].message.content)

out_tokens = resp.usage.completion_tokens
print(f"\n[INFO] {out_tokens} tokens in {dt:.2f}s "
      f"= {out_tokens / dt:.1f} tok/s")
