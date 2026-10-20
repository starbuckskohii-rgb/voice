"""Chấm toàn bộ audio trong ab_out/manifest.jsonl.

Mỗi file:
- asr: ReazonSpeech + SenseVoice (Whisper turbo bỏ vì hay rơi chữ ngay cả với audio chuẩn)
- keywords: cách đọc chuẩn (cột expected của ja_corpus) có xuất hiện trong cách đọc
  của chữ ASR ghi ra không (đọc bằng pyopenjtalk, bỏ qua khác biệt trường âm)
- particles: は/へ/を có bị đọc ha/he/wo không (model âm vị wav2vec2)
- speed: số mora / giây, để phát hiện bị cắt chữ hoặc nói dồn

Chạy (trong voxenv có sherpa-onnx):  python eval_all.py ab_out
Kết quả: ab_out/eval.jsonl + bảng tổng hợp theo biến thể.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pyopenjtalk
import sherpa_onnx
import soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ja_hard_detect import _normalize_pron  # noqa: E402
from particle_check import _coarse, check as particle_check  # noqa: E402

# Mặt chữ ASR ghi ra mà chắc chắn ứng với cách đọc đúng (pyopenjtalk đọc ngược các chữ
# này sai nên không dùng cách đọc được): 二十歳 -> はたち, 降りそう (trời mưa) -> ふりそう
ALIASES = {"はたち": ["二十歳", "ハタチ"], "ふりそう": ["降りそう", "ふりそう"], "ひとり": ["一人"],
           "ボックスシーピーエム": ["VOXCPM", "VoxCPM", "voxcpm", "BOXCPM"]}

M = os.path.join(HERE, "asr_models")
z = f"{M}/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01"
v = f"{M}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
RECS = {
    "Reazon": sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{z}/encoder-epoch-99-avg-1.onnx", decoder=f"{z}/decoder-epoch-99-avg-1.onnx",
        joiner=f"{z}/joiner-epoch-99-avg-1.onnx", tokens=f"{z}/tokens.txt", num_threads=4),
    "SenseV": sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{v}/model.onnx", tokens=f"{v}/tokens.txt", language="ja", use_itn=False, num_threads=4),
}


def asr(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample_poly(audio, 16000, sr).astype(np.float32)
    # ASR hụt âm đầu nếu audio bắt đầu ngay ở 0s -> đệm 0.5s lặng trước, 0.3s sau
    audio = np.concatenate([np.zeros(8000, np.float32), audio, np.zeros(4800, np.float32)])
    out = {}
    for name, rec in RECS.items():
        s = rec.create_stream()
        s.accept_waveform(16000, audio)
        rec.decode_stream(s)
        out[name] = s.result.text
    return out


def reading(text):
    return _normalize_pron(pyopenjtalk.g2p(text, kana=True)) if text.strip() else ""


def kw_reading(kw):
    return _normalize_pron("".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in kw))


def score_keywords(r):
    """Từ khoá đạt nếu: cách đọc của chữ ASR chứa nó, HOẶC chuỗi âm vị nghe được chứa
    nó, HOẶC ASR ghi đúng mặt chữ trong ALIASES. Tính lại được từ kết quả đã lưu."""
    readings = [reading(t) for t in r["asr"].values()]
    phon = "".join(_coarse(t) for t in r.get("phonemes", "").split())
    kw_ok = {}
    for kw in r["expected"].split():
        kw_ph = "".join(_coarse(t) for t in pyopenjtalk.g2p(kw).split() if t not in ("pau", "cl"))
        kw_ok[kw] = (any(kw_reading(kw) in x for x in readings) or (len(kw_ph) >= 4 and kw_ph in phon)
                     or any(a in t for a in ALIASES.get(kw, []) for t in r["asr"].values()))
    r["kw_ok"], r["kw_fail"] = kw_ok, [k for k, ok in kw_ok.items() if not ok]
    return r


def evaluate(row):
    texts = asr(row["file"])
    try:
        pc = particle_check(row["sentence"], row["file"])
    except Exception as e:  # noqa: BLE001
        pc = dict(particles=[], phonemes=f"ERR {e}", ratio=0)
    morae = len(_normalize_pron(pyopenjtalk.g2p(row["sentence"], kana=True)))
    return score_keywords(dict(row, asr=texts, particles=pc["particles"], phonemes=pc["phonemes"],
                               phone_ratio=pc["ratio"], mora_per_sec=round(morae / max(row["audio_sec"], 0.1), 1)))


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ab_out")
    rows = [json.loads(l) for l in open(os.path.join(out_dir, "manifest.jsonl"))]
    done = {}
    ev_path = os.path.join(out_dir, "eval.jsonl")
    if os.path.exists(ev_path):
        done = {json.loads(l)["file"]: json.loads(l) for l in open(ev_path)}
    results = []
    for row in rows:
        r = score_keywords(done[row["file"]]) if row["file"] in done else evaluate(row)
        if row["file"] not in done:
            with open(ev_path, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        results.append(r)

    agg = defaultdict(lambda: dict(files=0, kw=0, kw_fail=0, fast=0, **{f"{c}{k}": 0 for c in "はへを" for k in ("", "_bad")}))
    for r in results:
        a = agg[r["variant"]]
        a["files"] += 1
        a["kw"] += len(r["kw_ok"])
        a["kw_fail"] += len(r["kw_fail"])
        for p in r["particles"]:
            a[p["particle"]] += 1
            a[p["particle"] + "_bad"] += p["misread"]
        a["fast"] += r["mora_per_sec"] > 11  # tiếng Nhật đọc tự nhiên ~7-9 mora/giây
    print(f"{'biến thể':8} {'file':>4} {'từ khoá sai':>11} {'は→ha':>7} {'へ→he':>7} {'を→wo':>7} {'quá nhanh':>9}")
    for v_, a in sorted(agg.items()):
        print(f"{v_:8} {a['files']:>4} {a['kw_fail']:>5}/{a['kw']:<5} {a['は_bad']:>3}/{a['は']:<3} "
              f"{a['へ_bad']:>3}/{a['へ']:<3} {a['を_bad']:>3}/{a['を']:<3} {a['fast']:>6}")


if __name__ == "__main__":
    main()
