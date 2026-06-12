import base64, io, time, requests
from PIL import Image

def encode_image(path, max_side=896, quality=85):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1:                          # shrink only, never upscale
        img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}", img.size
  
IMG_PATH = "/home/mjrovai/Pictures/man_cat_dog.jpg"
MAX_SIZE = 512

uri, size = encode_image(IMG_PATH, max_side=MAX_SIZE)
print(f"sending image at {size[0]}x{size[1]}")

t0 = time.perf_counter()
r = requests.post(
    "http://localhost:8081/v1/chat/completions",
    json={"messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": uri}},
        ],
    }]},
    timeout=600,                           # don't give up during the long encode
)
dt = time.perf_counter() - t0
d = r.json()
t = d["timings"]

print(d["choices"][0]["message"]["content"])
print(f"\nimage+prompt : {t['prompt_n']:4d} tok in {t['prompt_ms']/1000:6.1f}s")
print(f"generation   : {t['predicted_n']:4d} tok in {t['predicted_ms']/1000:6.1f}s = {t['predicted_per_second']:.1f} tok/s")
print(f"wall         : {dt:.1f}s total")
