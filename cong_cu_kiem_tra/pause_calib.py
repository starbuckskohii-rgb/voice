"""Hiệu chỉnh find_unexpected_pauses: tỉ lệ câu bị gắn cờ theo nhóm giọng, và vị trí từng chỗ ngắt."""
import json, os, sys
import soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ja_process_v2 import JapaneseTextPreprocessor, find_unexpected_pauses, trim_silence, make_sensevoice_transcriber

pre = JapaneseTextPreprocessor()
tr = make_sensevoice_transcriber(os.path.join(HERE, "asr_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"))
rows = [json.loads(l) for l in open(os.path.join(HERE, "naturalness.jsonl"))]
groups = {}
for r in rows:
    if not (r["group"].startswith("Không giọng") or r["group"].startswith("Pipeline cũ")):
        groups.setdefault(r["group"], []).append((r["text"], r["file"]))
pj = [json.loads(l) for l in open(os.path.join(HERE, "jsut_out/manifest.jsonl"))]
groups["JSUT clone (AI)"] = [(r["sentence"], r["file"]) for r in pj if r["name"].endswith("_Fjsut")]
TH = [float(t) for t in sys.argv[1:]] or [0.15]
for g, items in groups.items():
    cnt = [0] * len(TH)
    for text, path in items:
        x, sr = sf.read(path, dtype="float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        x = trim_silence(x, sr)
        t = pre.process(text)
        heard = tr(x, sr)
        for i, th in enumerate(TH):
            found = find_unexpected_pauses(x, sr, t, heard, min_pause=th)
            cnt[i] += bool(found)
            if found and i == 0:
                print(f"   [{g}] {os.path.basename(path)}: {found}  | {text}")
    print(f"{g:30}" + "".join(f"  ≥{th:.2f}s: {c}/{len(items)}" for th, c in zip(TH, cnt)), flush=True)
