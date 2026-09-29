# Replicates the protocol of mjrovai.com/articles/slm-on-raspberry-pi-mtp on the Orange Pi Zero 3W:
# llama-server /completion, "Explain photosynthesis in 300 words.", n_predict 256, cache_prompt false, 3 runs.
import json, subprocess, time, urllib.request, os, statistics as st

H = os.path.expanduser
BIN = H("~/llama.cpp/build/bin/llama-server")
G = H("~/models/gemma-4-E2B-qat-it-GGUF")
Q = H("~/models/Qwen3.5-4B-MTP-GGUF")
MODELS = {
    "Gemma 4 E2B": dict(
        args=["--model", f"{G}/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf", "--flash-attn", "on", "--n-gpu-layers", "0",
              "--temp", "1.0", "--top-p", "0.95", "--top-k", "64"],
        draft=["--model-draft", f"{G}/mtp-gemma-4-E2B-it.gguf"],
        threads=[(2, 2), (3, 3), (4, 4), (2, 4), (2, 8)], ns=[0, 1, 2, 3, 4, 7]),
    "Qwen3.5 4B": dict(
        args=["--model", f"{Q}/Qwen3.5-4B-UD-Q4_K_XL.gguf", "--flash-attn", "on", "--kv-unified", "--jinja",
              "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0"],
        draft=["--spec-draft-n-min", "0"],
        threads=[(2, 2), (4, 4), (2, 4), (2, 8)], ns=[0, 1, 2, 3, 7]),
}
COMMON = ["--ctx-size", "8192", "--parallel", "1", "--reasoning", "off", "--reasoning-budget", "0",
          "--host", "127.0.0.1", "--port", "8099"]
URL = "http://127.0.0.1:8099"
BODY = {"prompt": "Explain photosynthesis in 300 words.", "n_predict": 256, "cache_prompt": False}


def rd(p):
    return int(open(p).read())


def post(body):
    r = urllib.request.Request(URL + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))["timings"]


out = open(H("~/bench_rpi_replica.jsonl"), "a")
for mname, m in MODELS.items():
    for (t, tb) in m["threads"]:
        for n in m["ns"]:
            if n == 0 and tb != t:
                continue  # tb only matters for multi-token (verify) batches
            cmd = [BIN] + m["args"] + COMMON + ["--threads", str(t), "--threads-batch", str(tb)]
            if n:
                cmd += m["draft"] + ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)]
            p = subprocess.Popen(cmd, stdout=open(H("~/srv_replica.log"), "w"), stderr=subprocess.STDOUT)
            for _ in range(600):
                try:
                    if json.load(urllib.request.urlopen(URL + "/health", timeout=2)).get("status") == "ok":
                        break
                except Exception:
                    pass
                time.sleep(1)
            post(dict(BODY, n_predict=16))  # warm-up
            runs = []
            for i in range(3):
                tm = post(BODY)
                acc = tm.get("draft_n_accepted") or 0
                runs.append(dict(tg=tm["predicted_per_second"], n=tm["predicted_n"], draft=tm.get("draft_n") or 0,
                                 acc=acc, mean_len=tm["predicted_n"] / max(1, tm["predicted_n"] - acc),
                                 temp=rd("/sys/class/thermal/thermal_zone0/temp") // 1000,
                                 f_big=rd("/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq") // 1000))
            row = dict(model=mname, t=t, tb=tb, n=n,
                       tg=round(st.mean(r["tg"] for r in runs), 2), sd=round(st.pstdev(r["tg"] for r in runs), 2),
                       mean_len=round(st.mean(r["mean_len"] for r in runs), 2),
                       acc_rate=round(sum(r["acc"] for r in runs) / max(1, sum(r["draft"] for r in runs)), 3),
                       max_temp=max(r["temp"] for r in runs), min_f_big=min(r["f_big"] for r in runs), runs=runs)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{mname:12} t={t} tb={tb} n={n}  {row['tg']:5.2f}+-{row['sd']:.2f} t/s  len={row['mean_len']}  "
                  f"acc={row['acc_rate']}  {row['max_temp']}C  {row['min_f_big']}MHz", flush=True)
            p.terminate()
            p.wait()
            time.sleep(15)
print("REPLICA_DONE", flush=True)
