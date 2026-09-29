import json, subprocess, time, urllib.request, os, sys
BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
MODELS = {
  "0.8B Q8": os.path.expanduser("~/models/Qwen3.5-0.8B-MTP-GGUF/Qwen3.5-0.8B-UD-Q8_K_XL.gguf"),
  "2B Q4_K_XL": os.path.expanduser("~/models/Qwen3.5-2B-MTP-GGUF/Qwen3.5-2B-UD-Q4_K_XL.gguf"),
}
PROMPTS = ["Write a Python function for binary search.",
           "Explain how a microcontroller reads an analog sensor.",
           "List ten uses of edge AI in agriculture."]
MODES = [("none", None), ("mtp-1", 1), ("mtp-2", 2), ("mtp-3", 3)]
URL = "http://127.0.0.1:8099"
def temp(): return int(open("/sys/class/thermal/thermal_zone0/temp").read())//1000
def req(prompt):
    body = {"messages":[{"role":"user","content":prompt}], "max_tokens":128, "temperature":0,
            "chat_template_kwargs":{"enable_thinking":False}}
    r = urllib.request.Request(URL+"/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r, timeout=900))
rows = []
for mname, mpath in MODELS.items():
    for mode, n in MODES:
        cmd = ["taskset","-c","6,7",BIN,"-m",mpath,"-t","2","-c","2048","--port","8099","-np","1","--reasoning","off"]
        if n: cmd += ["--spec-type","draft-mtp","--spec-draft-n-max",str(n)]
        p = subprocess.Popen(cmd, stdout=open(os.path.expanduser(f"~/srv_{mode}.log"),"w"), stderr=subprocess.STDOUT)
        for _ in range(300):
            try:
                if json.load(urllib.request.urlopen(URL+"/health", timeout=2)).get("status")=="ok": break
            except Exception: pass
            time.sleep(1)
        req("Hi")  # warm-up
        for i, pr in enumerate(PROMPTS):
            t = req(pr)["timings"]
            row = dict(model=mname, mode=mode, prompt=i+1, tg=round(t["predicted_per_second"],2), n=t["predicted_n"],
                       pp=round(t["prompt_per_second"],1), draft=t.get("draft_n"), acc=t.get("draft_n_accepted"), temp=temp())
            rows.append(row); print(json.dumps(row), flush=True)
        p.terminate(); p.wait()
json.dump(rows, open(os.path.expanduser("~/bench_mtp.json"),"w"), indent=1)
print("MTP_DONE")
