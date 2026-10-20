"""So sánh ngữ điệu + độ tự nhiên: giọng người thật (JSUT) vs các cấu hình VoxCPM2.

Chỉ số mỗi nhóm:
- UTMOS: điểm tự nhiên dự đoán (1-5) của model UTMOS22 strong
- trọng âm: tỉ lệ chỗ lên/xuống cao độ trong cụm trọng âm đi đúng chuẩn Tokyo (prosody_check)
- mora/giây và biên độ cao độ (semitone) để so nhịp và độ "phẳng" của giọng
"""

import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "models/SpeechMOS"))
from ja_hard_detect import _normalize_pron  # noqa: E402
from ja_process_v2 import trim_silence  # noqa: E402
from prosody_check import measure  # noqa: E402
from speechmos.utmos22.strong.model import UTMOS22Strong  # noqa: E402

import pyopenjtalk  # noqa: E402

utmos = UTMOS22Strong()
utmos.load_state_dict(torch.load(os.path.join(HERE, "models/utmos22_strong_step7459_v1.pt"), map_location="cpu", weights_only=True))
utmos.eval()


def human_set(n=40):
    out_dir = os.path.join(HERE, "human_ref/wav")
    os.makedirs(out_dir, exist_ok=True)
    rows = pq.read_table(os.path.join(HERE, "human_ref/jsut0.parquet")).to_pylist()
    picked = [r for r in rows if 12 <= len(r["transcription"]) <= 45][:n]
    items = []
    for i, r in enumerate(picked):
        p = os.path.join(out_dir, f"jsut_{i:03d}.flac")
        if not os.path.exists(p):
            open(p, "wb").write(r["audio"]["bytes"])
        items.append((r["transcription"], p))
    return items


def groups():
    g = {"Người thật (JSUT)": human_set()}
    final = [json.loads(l) for l in open(os.path.join(HERE, "final_out/manifest.jsonl"))]
    g["Cấu hình mới - giọng nữ"] = [(r["sentence"], r["file"]) for r in final if r["name"].endswith("_Ffemale")]
    g["Cấu hình mới - giọng nam"] = [(r["sentence"], r["file"]) for r in final if r["name"].endswith("_Fmale")]
    ab = [json.loads(l) for l in open(os.path.join(HERE, "ab_out/manifest.jsonl"))]
    g["Không giọng mẫu (A)"] = [(r["sentence"], r["file"]) for r in ab if r["variant"] == "A" and r["seed"] == 0]
    fx = [json.loads(l) for l in open(os.path.join(HERE, "fix_out/manifest.jsonl")) if '"id"' in l]
    g["Pipeline cũ (U)"] = ([(r["sentence"], r["file"]) for r in ab if r["variant"] == "U"]
                            + [(r["sentence"], r["file"]) for r in fx if r["name"].endswith("_Uprompt")])
    return g


def score(text, path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = trim_silence(x, sr)
    sf.write("/tmp/_nat.wav" if False else os.path.join(HERE, "audio/_nat.wav"), x, sr)
    x16 = resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x
    with torch.inference_mode():
        mos = float(utmos(torch.from_numpy(x16)[None], 16000))
    pr = measure(text, os.path.join(HERE, "audio/_nat.wav")) or dict(hit=0, total=0, f0_range_st=float("nan"))
    morae = len(_normalize_pron(pyopenjtalk.g2p(text, kana=True)))
    return dict(mos=mos, mora_s=morae / (len(x) / sr), **pr)


def main():
    out = os.path.join(HERE, "naturalness.jsonl")
    done = {}
    if os.path.exists(out):
        done = {(json.loads(l)["group"], json.loads(l)["file"]): json.loads(l) for l in open(out)}
    summary = {}
    for name, items in groups().items():
        rs = []
        for text, path in items:
            r = done.get((name, path))
            if r is None:
                r = dict(group=name, file=path, text=text, **score(text, path))
                with open(out, "a") as f:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rs.append(r)
        mos = np.array([r["mos"] for r in rs])
        hit, tot = sum(r["hit"] for r in rs), sum(r["total"] for r in rs)
        summary[name] = dict(n=len(rs), mos=f"{mos.mean():.2f} ± {mos.std():.2f}", accent=f"{100 * hit / max(tot, 1):.0f}% ({hit}/{tot})",
                             mora_s=f"{np.median([r['mora_s'] for r in rs]):.1f}",
                             f0_st=f"{np.nanmedian([r['f0_range_st'] for r in rs]):.1f}")
    print(f"{'Nhóm':28} {'n':>3} {'UTMOS':>12} {'Trọng âm đúng':>16} {'mora/s':>7} {'Biên độ F0 (st)':>16}")
    for k, s in summary.items():
        print(f"{k:28} {s['n']:>3} {s['mos']:>12} {s['accent']:>16} {s['mora_s']:>7} {s['f0_st']:>16}")


if __name__ == "__main__":
    main()
