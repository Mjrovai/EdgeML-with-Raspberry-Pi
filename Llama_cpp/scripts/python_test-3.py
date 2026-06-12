import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8081/v1", api_key="not-needed")

def ask(prompt, think=True):
    content = prompt if think else f"/no_think {prompt}"
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model="Gemma4",
        messages=[{"role": "user", "content": content}],
    )
    dt = time.perf_counter() - t0
    n = r.usage.completion_tokens
    print(f"thinking={'on ' if think else 'off'} | {n:3d} tok | "
          f"{dt:5.1f}s | {n/dt:.1f} tok/s")
    print("   →", r.choices[0].message.content.strip()[:80])

ask("What is the capital of Brazil?", think=True)
ask("What is the capital of Brazil?", think=False)
