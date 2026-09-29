# Peak memory of llama-server on the Orange Pi Zero 3W, per model/config, after one real request.
# Reports the process peak RSS (VmHWM, includes the mmap'ed GGUF pages) and the drop in the system's
# MemAvailable while the server is loaded. Usage: python3 bench_memory.py
import base64, json, subprocess, threading, time, urllib.request, os

H = os.path.expanduser
BIN = H("~/llama.cpp/build/bin/llama-server")
Q08, Q2, Q4, G = (H(p) for p in ("~/models/Qwen3.5-0.8B-MTP-GGUF", "~/models/Qwen3.5-2B-MTP-GGUF",
                                 "~/models/Qwen3.5-4B-MTP-GGUF", "~/models/gemma-4-E2B-qat-it-GGUF"))
MC = H("~/models")
CONFIGS = [
    ("Qwen3.5 0.8B Q8, text", ["-m", f"{Q08}/Qwen3.5-0.8B-UD-Q8_K_XL.gguf", "-c", "4096"], False),
    ("Qwen3.5 0.8B Q8, vision", ["-m", f"{Q08}/Qwen3.5-0.8B-UD-Q8_K_XL.gguf", "--mmproj", f"{Q08}/mmproj-F16.gguf", "-c", "4096"], True),
    ("Qwen3.5 2B Q4_K_XL, text", ["-m", f"{Q2}/Qwen3.5-2B-UD-Q4_K_XL.gguf", "-c", "4096"], False),
    ("Qwen3.5 2B Q4_K_XL, vision", ["-m", f"{Q2}/Qwen3.5-2B-UD-Q4_K_XL.gguf", "--mmproj", f"{Q2}/mmproj-F16.gguf", "-c", "4096"], True),
    ("MiniCPM5 1B Q4_K_M, text", ["-m", f"{MC}/MiniCPM5-1B-Q4_K_M/MiniCPM5-1B-Q4_K_M.gguf", "-c", "4096"], False),
    ("MiniCPM5 2B Q4_K_M, text", ["-m", f"{MC}/MiniCPM5-2B-Q4_K_M/MiniCPM5-2B-Q4_K_M.gguf", "-c", "4096"], False),
    ("Gemma 4 E2B QAT, text + MTP n=3", ["-m", f"{G}/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", "--model-draft", f"{G}/mtp-gemma-4-E2B-it.gguf",
                                        "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "-c", "8192", "--flash-attn", "on"], False),
    ("Gemma 4 E2B QAT, vision", ["-m", f"{G}/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", "--mmproj", f"{G}/mmproj-F16.gguf",
                                 "-c", "8192", "--flash-attn", "on"], True),
    ("Qwen3.5 4B Q4_K_XL, text + MTP n=3", ["-m", f"{Q4}/Qwen3.5-4B-UD-Q4_K_XL.gguf", "--spec-type", "draft-mtp", "--spec-draft-n-max", "3",
                                           "-c", "8192", "--flash-attn", "on", "--kv-unified"], False),
    ("Qwen3.5 4B Q4_K_XL, vision", ["-m", f"{Q4}/Qwen3.5-4B-UD-Q4_K_XL.gguf", "--mmproj", f"{Q4}/mmproj-F16.gguf",
                                    "-c", "8192", "--flash-attn", "on", "--kv-unified"], True),
]
URL = "http://127.0.0.1:8099"
b64 = base64.b64encode(open(H("~/test_cli_camera.jpg"), "rb").read()).decode()


def meminfo(key):
    for line in open("/proc/meminfo"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) // 1024  # MiB


def status(pid, key):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith(key + ":"):
            return int(line.split()[1]) // 1024  # MiB


def chat(vision):
    content = ([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": "Describe this image in one paragraph."}] if vision
               else "Explain photosynthesis in 300 words.")
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 256, "temperature": 0}
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))


# Drop the page cache is not possible without root, so MemAvailable (which counts reclaimable cache)
# is the fair system-level measure: it is what other programs could still get.
out = open(H("~/bench_memory.jsonl"), "a")
for name, args, vision in CONFIGS:
    avail0 = meminfo("MemAvailable")
    cmd = [BIN] + args + ["--jinja", "--reasoning", "off", "-t", "2", "-tb", "8", "--parallel", "1",
                          "--host", "127.0.0.1", "--port", "8099"]
    p = subprocess.Popen(cmd, stdout=open(H("~/srv_memory.log"), "w"), stderr=subprocess.STDOUT)
    for _ in range(900):
        try:
            if json.load(urllib.request.urlopen(URL + "/health", timeout=2)).get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(1)
    low = [avail0]
    stop = threading.Event()

    def watch():
        while not stop.wait(0.5):
            low.append(meminfo("MemAvailable"))
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    chat(vision)
    stop.set()
    th.join()
    row = dict(config=name, peak_rss_mib=status(p.pid, "VmHWM"), rss_anon_mib=status(p.pid, "RssAnon"),
               rss_file_mib=status(p.pid, "RssFile"), mem_available_before_mib=avail0,
               mem_available_min_mib=min(low), used_by_server_mib=avail0 - min(low),
               swap_used_mib=meminfo("SwapTotal") - meminfo("SwapFree"))
    out.write(json.dumps(row) + "\n")
    out.flush()
    print(json.dumps(row), flush=True)
    p.terminate()
    p.wait()
    time.sleep(10)
print("MEMORY_DONE", flush=True)
