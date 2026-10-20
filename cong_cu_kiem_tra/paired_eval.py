"""So cặp AI với người thật trên cùng câu, cùng giọng (JSUT).

Mỗi cặp: UTMOS hai bên, trọng âm đúng, biên độ F0, tỉ lệ độ dài AI/người, và tương
quan đường cao độ AI-người sau khi căn thời gian bằng DTW trên MFCC. Thêm: đọc đúng
chữ (ASR) và các biến thể cfg_value.
"""

import difflib
import json
import os
import re
import sys

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "models/SpeechMOS"))
from eval_all import asr  # noqa: E402
from ja_process_v2 import trim_silence  # noqa: E402
from prosody_check import measure  # noqa: E402
from speechmos.utmos22.strong.model import UTMOS22Strong  # noqa: E402

utmos = UTMOS22Strong()
utmos.load_state_dict(torch.load(os.path.join(HERE, "models/utmos22_strong_step7459_v1.pt"), map_location="cpu", weights_only=True))
utmos.eval()


def load16(path):
    x, sr = sf.read(path, dtype="float32")
    x = x.mean(axis=1) if x.ndim > 1 else x
    x = trim_silence(x, sr)
    return resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x


def stats(text, x):
    sf.write(os.path.join(HERE, "audio/_pair.wav"), x, 16000)
    with torch.inference_mode():
        mos = float(utmos(torch.from_numpy(x)[None], 16000))
    pr = measure(text, os.path.join(HERE, "audio/_pair.wav")) or dict(hit=0, total=0, f0_range_st=np.nan)
    return dict(mos=mos, dur=len(x) / 16000, **pr)


def f0_corr(a, b):
    """Tương quan log-F0 dọc đường DTW (MFCC) giữa hai bản ghi cùng câu."""
    hop = 160
    ma = librosa.feature.mfcc(y=a, sr=16000, n_mfcc=20, hop_length=hop)
    mb = librosa.feature.mfcc(y=b, sr=16000, n_mfcc=20, hop_length=hop)
    _, wp = librosa.sequence.dtw(X=ma, Y=mb, metric="cosine")
    fa, va, _ = librosa.pyin(a, fmin=60, fmax=500, sr=16000, frame_length=1024, hop_length=hop)
    fb, vb, _ = librosa.pyin(b, fmin=60, fmax=500, sr=16000, frame_length=1024, hop_length=hop)
    pairs = [(np.log(fa[i]), np.log(fb[j])) for i, j in wp[::-1]
             if i < len(fa) and j < len(fb) and va[i] and vb[j]]
    if len(pairs) < 20:
        return np.nan
    p = np.array(pairs)
    return float(np.corrcoef(p[:, 0], p[:, 1])[0, 1])


def text_ok(sentence, texts):
    norm = lambda t: re.sub(r"[、。！？「」…\s,.!?%％・]", "", t)  # noqa: E731
    return max(difflib.SequenceMatcher(None, norm(sentence), norm(t)).ratio() for t in texts.values())


def main():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "jsut_out/manifest.jsonl"))]
    out_path = os.path.join(HERE, "jsut_out/paired.jsonl")
    done = {json.loads(l)["name"]: json.loads(l) for l in open(out_path)} if os.path.exists(out_path) else {}
    res = []
    for r in rows:
        if r["name"] in done:
            res.append(done[r["name"]])
            continue
        ai = load16(r["file"])
        d = dict(name=r["name"], id=r["id"], sentence=r["sentence"], ai=stats(r["sentence"], ai),
                 asr=asr(r["file"]))
        d["text_match"] = text_ok(r["sentence"], d["asr"])
        if r.get("human"):
            hu = load16(r["human"])
            d["human"] = stats(r["sentence"], hu)
            d["f0_corr"] = f0_corr(ai, hu)
            d["dur_ratio"] = d["ai"]["dur"] / d["human"]["dur"]
        with open(out_path, "a") as f:
            f.write(json.dumps(d, ensure_ascii=False, default=float) + "\n")
        res.append(d)

    def agg(sel, label):
        if not sel:
            return
        ai_m = np.array([d["ai"]["mos"] for d in sel])
        hu = [d for d in sel if "human" in d]
        line = f"{label:22} n={len(sel):2}  UTMOS AI {ai_m.mean():.2f}"
        if hu:
            hu_m = np.array([d["human"]["mos"] for d in hu])
            line += (f" | người {hu_m.mean():.2f} | AI≥người {sum(d['ai']['mos'] >= d['human']['mos'] for d in hu)}/{len(hu)}"
                     f" | F0 tương quan {np.nanmean([d['f0_corr'] for d in hu]):.2f}"
                     f" | độ dài AI/người {np.median([d['dur_ratio'] for d in hu]):.2f}")
        ha, ta = sum(d["ai"]["hit"] for d in sel), sum(d["ai"]["total"] for d in sel)
        line += f" | trọng âm AI {100 * ha / max(ta, 1):.0f}%"
        if hu:
            hh, th = sum(d["human"]["hit"] for d in hu), sum(d["human"]["total"] for d in hu)
            line += f" (người {100 * hh / max(th, 1):.0f}%)"
        line += f" | F0 range AI {np.nanmedian([d['ai']['f0_range_st'] for d in sel]):.1f}st"
        if hu:
            line += f" (người {np.nanmedian([d['human']['f0_range_st'] for d in hu]):.1f}st)"
        line += f" | khớp chữ ≥0.9: {sum(d['text_match'] >= 0.9 for d in sel)}/{len(sel)}"
        print(line)

    agg([d for d in res if d["name"].endswith("_AI")], "JSUT cfg 2.0 (mặc định)")
    first5 = {f"jsut{i:02d}" for i in range(5)}
    agg([d for d in res if d["name"].endswith("_AI") and d["id"] in first5], "  5 câu đầu, cfg 2.0")
    agg([d for d in res if d["name"].endswith("_cfg15")], "  5 câu đầu, cfg 1.5")
    agg([d for d in res if d["name"].endswith("_cfg30")], "  5 câu đầu, cfg 3.0")
    agg([d for d in res if d["name"].endswith("_Fjsut")], "Bộ câu khó, giọng JSUT")


if __name__ == "__main__":
    main()
