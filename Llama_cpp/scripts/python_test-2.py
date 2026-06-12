import requests

r = requests.post(
    "http://localhost:8081/v1/chat/completions",
    json={"messages": [{"role": "user", "content": "What is the capital of Brazil?"}]},
)
data = r.json()
print(data["choices"][0]["message"]["content"])
print(data["timings"]["predicted_per_second"], "tok/s (server-measured)")
