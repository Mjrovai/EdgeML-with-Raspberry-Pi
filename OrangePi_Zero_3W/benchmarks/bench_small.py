# Qwen3.5 0.8B / 2B (Unsloth MTP GGUFs) on the Orange Pi Zero 3W or the Raspberry Pi 5, same protocol as
# mjrovai.com/articles/slm-on-raspberry-pi-mtp: /completion, "Explain photosynthesis in 300 words.",
# n_predict 256, cache_prompt false, 3 runs. Usage: python3 bench_small.py opi|rpi|unoq [qwen|minicpm]
import json, subprocess, sys, threading, time, urllib.request, os, statistics as st

H = os.path.expanduser
BOARD = sys.argv[1]
if BOARD == "opi":
    BIN_DIR = H("~/llama.cpp/build/bin")
    MODELS = {"Qwen3.5 0.8B Q8": H("~/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf"),
              "Qwen3.5 2B Q4_K_XL": H("~/models/Qwen3.5-2B-MTP-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf")}
    BENCH_THREADS = [2, 4, 8]
    THREADS = [(2, 2), (2, 4), (2, 8), (4, 4)]
elif BOARD == "unoq":
    BIN_DIR = H("~/llama.cpp-bench/build/bin")
    MODELS = {"Qwen3.5 0.8B Q8": H("~/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf"),
              "Qwen3.5 2B Q4_K_XL": H("~/models/Qwen3.5-2B-MTP-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf")}
    BENCH_THREADS = [1, 2, 3, 4]
    THREADS = [(2, 2), (3, 3), (4, 4), (2, 4)]
else:
    BIN_DIR = H("~/llama.cpp-bench/build/bin")
    MODELS = {"Qwen3.5 0.8B Q8": H("~/models/bench-mtp/Qwen3.5-0.8B-UD-Q8_K_XL.gguf"),
              "Qwen3.5 2B Q4_K_XL": H("~/models/bench-mtp/Qwen3.5-2B-UD-Q4_K_XL.gguf")}
    BENCH_THREADS = [1, 2, 3, 4]
    THREADS = [(2, 2), (3, 3), (4, 4), (2, 4)]
NS = [0, 1, 2, 3, 7]
SET = sys.argv[2] if len(sys.argv) > 2 else "qwen"
if SET == "minicpm":  # openbmb/MiniCPM5-*-GGUF, no MTP heads: plain decode only
    MODELS = {f"MiniCPM5 {s} Q4_K_M": H(f"~/models/MiniCPM5-{s}-Q4_K_M/MiniCPM5-{s}-Q4_K_M.gguf") for s in ("1B", "2B")}
    NS = [0]
    # a raw prompt without the chat template sometimes ends at the first token; force 256 tokens
    BODY_EXTRA = {"ignore_eos": True}
SERVER_ONLY = len(sys.argv) > 3 and sys.argv[3] == "server-only"
SAMPLING = ["--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0"]
COMMON = ["--ctx-size", "8192", "--parallel", "1", "--flash-attn", "on", "--kv-unified", "--jinja",
          "--reasoning", "off", "--reasoning-budget", "0", "--host", "127.0.0.1", "--port", "8099"]
URL = "http://127.0.0.1:8099"
BODY = {"prompt": "Explain photosynthesis in 300 words.", "n_predict": 256, "cache_prompt": False}
BODY.update(globals().get("BODY_EXTRA", {}))
OUT = open(H(f"~/bench_small_{BOARD}.jsonl" if SET == "qwen" else
             f"~/bench_{SET}_{BOARD}{'_server' if SERVER_ONLY else ''}.jsonl"), "a")


def health():
    """Temperature and throttle state. The Raspberry Pi reports throttling via vcgencmd;
    the Orange Pi has no equivalent, so its big-core clock is recorded instead."""
    if BOARD == "unoq":
        return dict(temp=cpu_temp(), f_mhz=int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read()) // 1000)
    if BOARD == "rpi":
        t = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True).stdout
        th = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout
        return dict(temp=float(t.split("=")[1].split("'")[0]), throttled=th.strip().split("=")[1])
    temp = int(open("/sys/class/thermal/thermal_zone0/temp").read()) // 1000
    f = int(open("/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq").read()) // 1000
    return dict(temp=temp, f_big_mhz=f)


def zone(name):
    for z in sorted(os.listdir("/sys/class/thermal")):
        if z.startswith("thermal_zone") and open(f"/sys/class/thermal/{z}/type").read().strip() == name:
            return f"/sys/class/thermal/{z}/temp"


CPU_TEMP = zone("cpuss0-thermal") if BOARD == "unoq" else None


def cpu_temp():
    return int(open(CPU_TEMP).read()) / 1000


class Sampler:
    """UNO Q only: samples CPU temperature and clock every second *while* a request runs,
    so a throttled run shows up as a lowered busy-core clock."""
    def __init__(self):
        self.temps, self.freqs, self.stop = [], [], threading.Event()
        self.t = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.wait(1.0):
            self.temps.append(cpu_temp())
            self.freqs.append(max(int(open(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq").read())
                                  for c in range(4)) // 1000)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.t.join()

    def result(self):
        return dict(max_temp=max(self.temps, default=None), min_busy_mhz=min(self.freqs, default=None))


def cool_down(limit=55, timeout=300):
    if BOARD != "unoq":
        return time.sleep(15)
    t0 = time.time()
    while cpu_temp() > limit and time.time() - t0 < timeout:
        time.sleep(5)


def post(body):
    r = urllib.request.Request(URL + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))["timings"]


def emit(row):
    OUT.write(json.dumps(row) + "\n")
    OUT.flush()
    print(json.dumps({k: v for k, v in row.items() if k != "runs"}), flush=True)


# 1. llama-bench pp512 / tg128
for mname, mpath in ({} if SERVER_ONLY else MODELS).items():
    for t in BENCH_THREADS:
        res = subprocess.run([f"{BIN_DIR}/llama-bench", "-m", mpath, "-t", str(t), "-p", "512", "-n", "128",
                              "-r", "3", "-o", "json"], capture_output=True, text=True)
        for r in json.loads(res.stdout):
            test = "pp512" if r["n_prompt"] else "tg128"
            emit(dict(kind="llama-bench", board=BOARD, model=mname, t=t, test=test,
                      ts=round(r["avg_ts"], 2), sd=round(r["stddev_ts"], 2), **health()))
        cool_down()

# 2. llama-server MTP sweep (article protocol)
for mname, mpath in MODELS.items():
    for (t, tb) in THREADS:
        for n in NS:
            if n == 0 and tb != t:
                continue  # tb only matters for multi-token (verify) batches
            cmd = [f"{BIN_DIR}/llama-server", "--model", mpath] + SAMPLING + COMMON + \
                  ["--threads", str(t), "--threads-batch", str(tb)]
            if n:
                cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n), "--spec-draft-n-min", "0"]
            p = subprocess.Popen(cmd, stdout=open(H("~/srv_small.log"), "w"), stderr=subprocess.STDOUT)
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
                if BOARD == "unoq":
                    with Sampler() as smp:
                        tm = post(BODY)
                    extra = smp.result()
                else:
                    tm = post(BODY)
                    extra = health()
                acc = tm.get("draft_n_accepted") or 0
                runs.append(dict(tg=tm["predicted_per_second"], n=tm["predicted_n"], draft=tm.get("draft_n") or 0,
                                 acc=acc, mean_len=tm["predicted_n"] / max(1, tm["predicted_n"] - acc), **extra))
            emit(dict(kind="mtp", board=BOARD, model=mname, t=t, tb=tb, n=n,
                      tg=round(st.mean(r["tg"] for r in runs), 2), sd=round(st.pstdev(r["tg"] for r in runs), 2),
                      mean_len=round(st.mean(r["mean_len"] for r in runs), 2),
                      acc_rate=round(sum(r["acc"] for r in runs) / max(1, sum(r["draft"] for r in runs)), 3),
                      runs=runs))
            p.terminate()
            p.wait()
            cool_down()
print("SMALL_DONE", flush=True)
