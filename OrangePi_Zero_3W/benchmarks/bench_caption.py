# Image-captioning latency on the Orange Pi Zero 3W, Raspberry Pi 5, and UNO Q.
# llama-server --mmproj (Unsloth mmproj-F16), --image-max-tokens 256, same photo on every board,
# chat completion "Describe this image in one paragraph.", max_tokens 128, temperature 0, 3 runs.
# Usage: python3 bench_caption.py opi|rpi|unoq
import base64, json, re, subprocess, sys, time, urllib.request, os, statistics as st

H = os.path.expanduser
BOARD = sys.argv[1]
IMG = H("~/test_cli_camera.jpg")
if BOARD == "opi":
    BIN = H("~/llama.cpp/build/bin/llama-server")
    M = {"Qwen3.5 0.8B Q8": ("~/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf", "~/models/Qwen3.5-0.8B-MTP-GGUF/mmproj-F16.gguf"),
         "Qwen3.5 2B Q4_K_XL": ("~/models/Qwen3.5-2B-MTP-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf", "~/models/Qwen3.5-2B-MTP-GGUF/mmproj-F16.gguf"),
         "Qwen3.5 4B Q4_K_XL": ("~/models/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf", "~/models/Qwen3.5-4B-MTP-GGUF/mmproj-F16.gguf"),
         "Gemma 4 E2B QAT": ("~/models/gemma-4-E2B-qat-it-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", "~/models/gemma-4-E2B-qat-it-GGUF/mmproj-F16.gguf")}
    THREADS = [(2, 8), (8, 8)]
elif BOARD == "rpi":
    BIN = H("~/llama.cpp-bench/build/bin/llama-server")
    M = {"Qwen3.5 0.8B Q8": ("~/models/bench-mtp/Qwen3.5-0.8B-UD-Q8_K_XL.gguf", "~/models/bench-mtp/mmproj-0.8B-F16.gguf"),
         "Qwen3.5 2B Q4_K_XL": ("~/models/bench-mtp/Qwen3.5-2B-UD-Q4_K_XL.gguf", "~/models/bench-mtp/mmproj-2B-F16.gguf"),
         "Qwen3.5 4B Q4_K_XL": ("~/models/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf", "~/models/Qwen3.5-4B-MTP-GGUF/mmproj-F16.gguf"),
         "Gemma 4 E2B QAT": ("~/models/gemma-4-E2B-qat-it-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", "~/models/gemma-4-E2B-qat-it-GGUF/mmproj-F16.gguf")}
    THREADS = [(4, 4), (2, 4)]
else:
    BIN = H("~/llama.cpp-bench/build/bin/llama-server")
    M = {"Qwen3.5 0.8B Q8": ("~/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf", "~/models/Qwen3.5-0.8B-MTP-GGUF/mmproj-F16.gguf"),
         "Qwen3.5 2B Q4_K_XL": ("~/models/Qwen3.5-2B-MTP-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf", "~/models/Qwen3.5-2B-MTP-GGUF/mmproj-F16.gguf")}
    THREADS = [(4, 4)]
URL = "http://127.0.0.1:8099"
LOG = H("~/srv_caption.log")
b64 = base64.b64encode(open(IMG, "rb").read()).decode()
BODY = {"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "Describe this image in one paragraph."}]}],
        "max_tokens": 128, "temperature": 0, "cache_prompt": False}
OUT = open(H(f"~/bench_caption_{BOARD}.jsonl"), "a")


def post():
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(BODY).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(r, timeout=3600))
    return time.time() - t0, resp


def encode_ms():
    """Last 'image slice encoded in X ms' reported by the server (vision encoder only)."""
    hits = re.findall(r"encoded in (\d+) ms", open(LOG, errors="replace").read())
    return int(hits[-1]) if hits else None


for mname, (model, mmproj) in M.items():
    for (t, tb) in THREADS:
        cmd = [BIN, "-m", H(model), "--mmproj", H(mmproj), "--jinja", "-c", "4096", "--image-max-tokens", "256",
               "--reasoning", "off", "-t", str(t), "-tb", str(tb), "--parallel", "1", "--host", "127.0.0.1", "--port", "8099"]
        p = subprocess.Popen(cmd, stdout=open(LOG, "w"), stderr=subprocess.STDOUT)
        for _ in range(900):
            try:
                if json.load(urllib.request.urlopen(URL + "/health", timeout=2)).get("status") == "ok":
                    break
            except Exception:
                pass
            time.sleep(1)
        post()  # warm-up (loads pages of both GGUFs into the page cache)
        runs = []
        for i in range(3):
            wall, resp = post()
            tm = resp["timings"]
            runs.append(dict(wall_s=round(wall, 2), encode_ms=encode_ms(), prompt_n=tm["prompt_n"],
                             prompt_ms=round(tm["prompt_ms"]), gen_n=tm["predicted_n"],
                             gen_tps=round(tm["predicted_per_second"], 2),
                             caption=resp["choices"][0]["message"]["content"]))
        row = dict(board=BOARD, model=mname, t=t, tb=tb,
                   wall_s=round(st.mean(r["wall_s"] for r in runs), 2),
                   encode_s=round(st.mean(r["encode_ms"] for r in runs if r["encode_ms"]) / 1000, 2) if any(r["encode_ms"] for r in runs) else None,
                   prompt_s=round(st.mean(r["prompt_ms"] for r in runs) / 1000, 2),
                   gen_tps=round(st.mean(r["gen_tps"] for r in runs), 2), runs=runs)
        OUT.write(json.dumps(row) + "\n")
        OUT.flush()
        print(json.dumps({k: v for k, v in row.items() if k != "runs"}), flush=True)
        p.terminate()
        p.wait()
        time.sleep(30)
print("CAPTION_DONE", flush=True)
