"""Photo -> line-art conversion via the BFL FLUX.2 API (image-editing endpoint).

    BFL_API_KEY=... python -m unicasso.lineart.bfl_lineart <photo-root> --out <line-root> \
        [--model flux-2-pro] [--concurrency 4] [--dry-run]

Walks the photo tree, submits each image with the line-art instruction prompt
(unicasso/lineart/prompts/flux_line_prompt.txt by default), polls, writes the result PNG into a mirrored
tree. RESUME-SAFE (skips outputs that exist); failures logged and skipped. Cost note:
BFL bills by megapixel -- inputs are downscaled to ~1MP and output matches that size,
which is plenty for asciify targets (they get resized to the render canvas anyway).
"""
import argparse
import base64
import io
import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
API = "https://api.bfl.ai/v1"
PROMPT_DEFAULT = os.path.join(os.path.dirname(__file__), "prompts", "flux_line_prompt.txt")


def collect(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if os.path.splitext(fn)[1].lower() in EXTS:
                out.append(os.path.join(dp, fn))
    return out


def _post(url, key, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"x-key": key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def _get(url, key):
    req = urllib.request.Request(url, headers={"x-key": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def convert_one(img_path, out_path, key, model, prompt, max_side, timeout_s=240):
    im = Image.open(img_path)
    im = ImageOps.exif_transpose(im).convert("RGB")   # phone JPEGs: rotation lives in EXIF
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    w = (im.width // 32) * 32
    h = (im.height // 32) * 32
    task = _post(f"{API}/{model}", key, dict(
        prompt=prompt,
        input_image=base64.b64encode(buf.getvalue()).decode(),
        width=max(w, 64), height=max(h, 64),
        output_format="png", safety_tolerance=2))
    t0 = time.time()
    while True:
        time.sleep(2.0)
        res = _get(task["polling_url"], key)
        status = res.get("status")
        if status == "Ready":
            sample = res["result"]["sample"]
            with urllib.request.urlopen(sample, timeout=120) as r:
                data = r.read()
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(data)
            return True, ""
        if status in ("Error", "Content Moderated", "Request Moderated", "Task not found"):
            return False, f"{status}: {res.get('details', '')}"
        if time.time() - t0 > timeout_s:
            return False, "poll timeout"


def main():
    ap = argparse.ArgumentParser(description="BFL FLUX.2 photo -> lineart batch")
    ap.add_argument("root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="flux-2-pro")
    ap.add_argument("--prompt-file", default=PROMPT_DEFAULT)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    key = os.environ.get("BFL_API_KEY")
    if not key and not args.dry_run:
        raise SystemExit("set BFL_API_KEY (from your BFL profile)")
    with open(args.prompt_file, encoding="utf-8") as f:
        prompt = f.read().strip()

    imgs = collect(args.root)
    jobs = []
    for p in imgs:
        rel = os.path.splitext(os.path.relpath(p, args.root))[0] + "_line.png"
        o = os.path.join(args.out, rel)
        if not os.path.exists(o):
            jobs.append((p, o))
    print(f"{len(imgs)} photos, {len(imgs) - len(jobs)} already converted, {len(jobs)} to run")
    if args.dry_run:
        for p, o in jobs[:8]:
            print(f"  {p} -> {o}")
        return

    lock, done, failures = threading.Lock(), [0], []

    def run(job):
        p, o = job
        try:
            ok, msg = convert_one(p, o, key, args.model, prompt, args.max_side)
        except Exception as e:                                # network hiccup etc.
            ok, msg = False, repr(e)
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(jobs)}] {'ok' if ok else 'FAIL ' + msg}  {os.path.basename(p)}",
                  flush=True)
            if not ok:
                failures.append((p, msg))

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(run, jobs))
    print(f"\ndone: {len(jobs) - len(failures)} ok, {len(failures)} failed")
    for p, m in failures:
        print("  failed:", p, "--", m)


if __name__ == "__main__":
    main()
