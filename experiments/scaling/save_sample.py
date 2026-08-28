"""
Quick one-shot: generate a single image from the batch scaling experiment prompt and save it.
"""
import base64, json, os, urllib.request

BASE_URL = "http://127.0.0.1:7860"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "images")
os.makedirs(OUT_DIR, exist_ok=True)

payload = {
    "prompt": "a serene mountain lake at sunrise, photorealistic, fine details",
    "negative_prompt": "",
    "width": 1024,
    "height": 1024,
    "steps": 8,
    "cfg_scale": 1.0,
    "sampler_name": "Euler",
    "batch_size": 1,
    "n_iter": 1,
    "seed": 12345,
}
body = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    BASE_URL + "/sdapi/v1/txt2img",
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=300) as resp:
    data = json.loads(resp.read().decode("utf-8"))

img_b64 = data["images"][0]
img_bytes = base64.b64decode(img_b64.split(",", 1)[0])
out_path = os.path.join(OUT_DIR, "batch_scaling_sample.png")
with open(out_path, "wb") as f:
    f.write(img_bytes)
print(f"Saved: {out_path}  ({len(img_bytes)} bytes)")