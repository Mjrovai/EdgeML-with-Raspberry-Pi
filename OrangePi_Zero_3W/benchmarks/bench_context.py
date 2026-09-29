# Context-length effects on the Orange Pi Zero 3W.
# A) llama-bench at context depth 0 / 4096 / 16384: prompt processing (-t 8) and generation (-t 2).
# B) Multi-turn prompt caching with an agent-style ~2k-token system prompt: how many tokens the
#    server re-processes on turn 2, with default checkpoints and with --checkpoint-min-step 256.
# Usage: python3 bench_context.py
import json, subprocess, time, urllib.request, os

H = os.path.expanduser
BIN = H("~/llama.cpp/build/bin")
MODELS = {
    "Qwen3.5 2B Q4_K_XL": H("~/models/Qwen3.5-2B-MTP-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf"),
    "MiniCPM5 2B Q4_K_M": H("~/models/MiniCPM5-2B-Q4_K_M/MiniCPM5-2B-Q4_K_M.gguf"),
    "Gemma 4 E2B QAT": H("~/models/gemma-4-E2B-qat-it-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf"),
    "Qwen3.5 4B Q4_K_XL": H("~/models/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf"),
}
DEPTHS = "0,4096,16384"
OUT = open(H("~/bench_context.jsonl"), "a")


def emit(row):
    OUT.write(json.dumps(row) + "\n")
    OUT.flush()
    print(json.dumps({k: v for k, v in row.items() if k != "answer"}), flush=True)


# A) throughput vs. context depth
for name, path in MODELS.items():
    for test, t in (("pp512", ["-p", "512", "-n", "0", "-t", "8"]), ("tg128", ["-p", "0", "-n", "128", "-t", "2"])):
        res = subprocess.run([f"{BIN}/llama-bench", "-m", path, "-fa", "1", "-d", DEPTHS, "-r", "1", "-o", "json"] + t,
                             capture_output=True, text=True)
        try:
            for r in json.loads(res.stdout):
                emit(dict(part="depth", model=name, test=test, depth=r["n_depth"], ts=round(r["avg_ts"], 2)))
        except Exception:
            emit(dict(part="depth", model=name, test=test, error=res.stderr[-400:]))
        time.sleep(20)

# B) prompt caching across turns
TOOLS = [{"type": "function", "function": {
    "name": f"sensor_{i}", "description": f"Read sensor {i} of the greenhouse node and return the latest value "
                                          f"with its unit, timestamp, calibration status, and alarm thresholds.",
    "parameters": {"type": "object", "properties": {
        "node_id": {"type": "string", "description": "Identifier of the edge node, for example 'gh-01'."},
        "window_s": {"type": "integer", "description": "Averaging window in seconds (1 to 3600)."},
        "unit": {"type": "string", "enum": ["metric", "imperial"], "description": "Unit system for the value."}},
        "required": ["node_id"]}}} for i in range(24)]
SYSTEM = ("You are an agent running on an edge device in a greenhouse. Use the tools to read sensors before "
          "answering. Keep answers short. ") * 4
URL = "http://127.0.0.1:8099"


def chat(messages):
    body = {"messages": messages, "tools": TOOLS, "max_tokens": 48, "temperature": 0}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(r, timeout=3600))
    return time.time() - t0, resp


for name, path in MODELS.items():
    for label, extra in (("default", []), ("checkpoint-min-step 256", ["--checkpoint-min-step", "256"])):
        cmd = [f"{BIN}/llama-server", "-m", path, "-c", "8192", "-fa", "on", "--jinja", "--reasoning", "off",
               "-t", "2", "-tb", "8", "--parallel", "1", "--host", "127.0.0.1", "--port", "8099"] + extra
        p = subprocess.Popen(cmd, stdout=open(H("~/srv_context.log"), "w"), stderr=subprocess.STDOUT)
        for _ in range(900):
            try:
                if json.load(urllib.request.urlopen(URL + "/health", timeout=2)).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What is the temperature at node gh-01?"}]
        turns = []
        for turn in (1, 2, 3):
            wall, resp = chat(msgs)
            tm = resp["timings"]
            turns.append(dict(turn=turn, prompt_n=tm["prompt_n"], prompt_s=round(tm["prompt_ms"] / 1000, 1),
                              cache_n=tm.get("cache_n"), wall_s=round(wall, 1)))
            msg = resp["choices"][0]["message"]
            msgs.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})})
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    msgs.append({"role": "tool", "tool_call_id": tc.get("id", "0"), "content": '{"value": 24.3, "unit": "C"}'})
            msgs.append({"role": "user", "content": "And the humidity at node gh-02?" if turn == 1 else "Thanks. Any alarms?"})
        emit(dict(part="cache", model=name, config=label, turns=turns))
        p.terminate()
        p.wait()
        time.sleep(20)
print("CONTEXT_DONE", flush=True)
