import json, subprocess, time, urllib.request, os
BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
MODELS = {
  "0.8B Q8": os.path.expanduser("~/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf"),
  "2B Q4_K_XL": os.path.expanduser("~/models/Qwen3.5-2B-MTP-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf"),
}
CONFIGS = {
  "A ref: taskset A76, -t 2":                 (["taskset","-c","6,7"], ["-t","2"]),
  "B taskset A76, -t 2 -tb 8":                (["taskset","-c","6,7"], ["-t","2","-tb","8"]),
  "C -t 2 on A76 strict, -tb 8 all strict":   ([], ["-t","2","-C","0xC0","--cpu-strict","1","-tb","8","-Cb","0xFF","--cpu-strict-batch","1"]),
  "D -t 2 on A76 strict, -tb 8 all loose":    ([], ["-t","2","-C","0xC0","--cpu-strict","1","-tb","8","-Cb","0xFF","--cpu-strict-batch","0"]),
  "E -t 2 -tb 8, no affinity":                ([], ["-t","2","-tb","8"]),
}
URL = "http://127.0.0.1:8099"
base = ("Edge AI moves machine learning inference from the cloud to small devices close to the sensors. "
        "This reduces latency, saves bandwidth, and keeps private data on the device. ")
PROMPT = base * 16 + "\nSummarize the text above in detail:"
def temp(): return int(open("/sys/class/thermal/thermal_zone0/temp").read())//1000
def req(n_predict=128):
    body = {"prompt": PROMPT, "n_predict": n_predict, "temperature": 0, "cache_prompt": False, "ignore_eos": True}
    r = urllib.request.Request(URL+"/completion", json.dumps(body).encode(), {"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))["timings"]
rows = []
for mname, mpath in MODELS.items():
    for cname, (pre, args) in CONFIGS.items():
        cmd = pre + [BIN,"-m",mpath,"-c","2048","--port","8099","-np","1"] + args
        p = subprocess.Popen(cmd, stdout=open(os.path.expanduser("~/srv_aff.log"),"w"), stderr=subprocess.STDOUT)
        for _ in range(300):
            try:
                if json.load(urllib.request.urlopen(URL+"/health", timeout=2)).get("status")=="ok": break
            except Exception: pass
            time.sleep(1)
        req(8)  # warm-up
        for i in range(3):
            t = req()
            row = dict(model=mname, config=cname, rep=i+1, pp=round(t["prompt_per_second"],2), pn=t["prompt_n"],
                       tg=round(t["predicted_per_second"],2), n=t["predicted_n"], temp=temp())
            rows.append(row); print(json.dumps(row), flush=True)
        p.terminate(); p.wait(); time.sleep(20)  # cool-down between configs
json.dump(rows, open(os.path.expanduser("~/bench_affinity.json"),"w"), indent=1)
print("AFF_DONE")
