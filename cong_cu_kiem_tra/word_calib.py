"""Hiệu chỉnh missing_words: tỉ lệ câu bị báo thiếu từ theo nhóm (người thật đọc đúng -> báo là báo nhầm)."""
import json, os, sys
import soundfile as sf
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ja_process_v2 import JapaneseTextPreprocessor, missing_words, trim_silence, make_sensevoice_transcriber

pre = JapaneseTextPreprocessor()
tr = make_sensevoice_transcriber(os.path.join(HERE, "asr_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"))
groups = {}
for l in open(os.path.join(HERE, "naturalness.jsonl")):
    r = json.loads(l)
    if not (r["group"].startswith("Không giọng") or r["group"].startswith("Pipeline cũ")):
        groups.setdefault(r["group"], []).append((r["text"], r["file"]))
pj = [json.loads(l) for l in open(os.path.join(HERE, "jsut_out/manifest.jsonl"))]
groups["JSUT clone (AI)"] = [(r["sentence"], r["file"]) for r in pj if r["name"].endswith("_Fjsut")]
extra = sys.argv[1:]
if extra:
    groups["Thêm"] = [json.loads(l) for l in open(extra[0])]
    groups["Thêm"] = [(r["sentence"], r["file"]) for r in groups["Thêm"]]
for g, items in groups.items():
    bad = 0
    for text, path in items:
        x, sr = sf.read(path, dtype="float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        x = trim_silence(x, sr)
        heard = tr(x, sr)
        miss = missing_words(pre.process(text), heard)
        bad += bool(miss)
        if miss:
            print(f"   [{g}] {os.path.basename(path)}: thiếu {miss} | {text} | nghe: {heard}")
    print(f"{g:30} báo thiếu từ: {bad}/{len(items)}", flush=True)
