"""Đo trọng âm cao độ (pitch accent) tiếng Nhật của một file audio so với chuẩn Tokyo.

1. pyopenjtalk cho nhãn đầy đủ: mỗi mora thuộc cụm trọng âm nào (accent phrase), vị
   trí trong cụm, kiểu trọng âm -> mẫu cao/thấp chuẩn (L/H) từng mora.
2. Căn chỉnh CTC (wav2vec2 âm vị) để biết mỗi âm vị nằm ở khung thời gian nào.
3. Đo F0 (librosa.pyin), lấy trung vị log-F0 mỗi mora.
4. Điểm = tỉ lệ chỗ chuyển cao/thấp (L→H, H→L) trong cụm mà audio đi đúng chiều.

Điểm này chỉ có nghĩa khi so với giọng người thật đo cùng cách (corpus JSUT).
"""

import json
import os
import re

import librosa
import numpy as np
import pyopenjtalk
import soundfile as sf
import torch
import torchaudio.functional as AF
from scipy.signal import resample_poly
from transformers import Wav2Vec2ForCTC

HERE = os.path.dirname(os.path.abspath(__file__))
_M = os.path.join(HERE, "models/w2v2-phoneme")
_model = Wav2Vec2ForCTC.from_pretrained(_M).eval()
_vocab = json.load(open(os.path.join(_M, "vocab.json")))

# âm vị pyopenjtalk -> token IPA kiểu espeak tiếng Nhật trong vocab của model
_MAP = {
    "a": ["a"], "i": ["i"], "u": ["ɯ"], "e": ["e"], "o": ["o"],
    "A": ["a"], "I": ["i"], "U": ["ɯ"], "E": ["e"], "O": ["o"],
    "N": ["ɴ"], "k": ["k"], "g": ["ɡ"], "s": ["s"], "z": ["z"], "sh": ["ɕ"], "j": ["dʑ"],
    "t": ["t"], "ch": ["tɕ"], "ts": ["ts"], "d": ["d"], "n": ["n"], "h": ["h"], "f": ["ɸ"],
    "b": ["b"], "p": ["p"], "m": ["m"], "y": ["j"], "r": ["ɾ"], "w": ["w"], "v": ["v"],
    "ky": ["k", "j"], "gy": ["ɡ", "j"], "ny": ["ɲ"], "hy": ["ç"], "by": ["b", "j"], "py": ["p", "j"],
    "my": ["m", "j"], "ry": ["ɾ", "j"], "dy": ["d", "j"], "ty": ["t", "j"],
}
_LAB = re.compile(r"\^(?P<p2>[^-]+)-(?P<p3>[^+]+)\+.*?/A:(?P<a1>[^+]+)\+(?P<a2>[^+]+)\+(?P<a3>[^/]+)/"
                  r".*?/F:(?P<f1>[^_]+)_(?P<f2>[^#]+)#.*?@(?P<f5>[^_]+)_.*?/I:.*?@(?P<i3>[^+]+)\+")


def accent_moras(text):
    """[(phrase_key, mora_pos, expected 'H'/'L', [token ids])] theo thứ tự."""
    moras = []
    for lab in pyopenjtalk.extract_fullcontext(text):
        m = _LAB.search(lab)
        if not m or m["p3"] in ("sil", "pau", "cl") or m["a2"] == "xx":
            continue
        toks = _MAP.get(m["p3"])
        if not toks or any(t not in _vocab for t in toks):
            continue
        n, acc, pos = int(m["f1"]), int(m["f2"]), int(m["a2"])
        if acc == 0:
            hl = "L" if pos == 1 else "H"
        elif acc == 1:
            hl = "H" if pos == 1 else "L"
        else:
            hl = "L" if pos == 1 else ("H" if pos <= acc else "L")
        key = (m["i3"], m["f5"])
        ids = [_vocab[t] for t in toks]
        if moras and moras[-1][0] == key and moras[-1][1] == pos:
            moras[-1][3].extend(ids)
        else:
            moras.append([key, pos, hl, ids])
    return moras


def _load16k(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x


def measure(text, path):
    moras = accent_moras(text)
    audio = _load16k(path)
    with torch.inference_mode():
        a = (audio - audio.mean()) / (audio.std() + 1e-7)
        logits = _model(torch.from_numpy(a)[None]).logits
    emis = torch.log_softmax(logits, -1)
    targets = [t for mo in moras for t in mo[3]]
    if not targets or emis.shape[1] < len(targets):
        return None
    ali, _ = AF.forced_align(emis, torch.tensor([targets], dtype=torch.int32), blank=0)
    ali = ali[0].tolist()
    # khung (20 ms) -> chỉ số token theo thứ tự xuất hiện
    tok_frames, k, prev = [[] for _ in targets], -1, 0
    for f, t in enumerate(ali):
        if t != 0 and (t != prev or k < 0):
            k += 1
        if t != 0 and 0 <= k < len(targets):
            tok_frames[k].append(f)
        prev = t
    f0, voiced, _ = librosa.pyin(audio, fmin=60, fmax=500, sr=16000, frame_length=1024, hop_length=320)
    logf0 = np.where(voiced, np.log(np.nan_to_num(f0, nan=1.0)), np.nan)
    i, mora_f0 = 0, []
    for mo in moras:
        frames = [f for j in range(i, i + len(mo[3])) for f in tok_frames[j]]
        i += len(mo[3])
        vals = [logf0[f] for f in frames if f < len(logf0) and not np.isnan(logf0[f])]
        mora_f0.append(np.median(vals) if vals else np.nan)
    hit = tot = 0
    for j in range(1, len(moras)):
        if moras[j][0] != moras[j - 1][0] or moras[j][2] == moras[j - 1][2]:
            continue
        a_, b_ = mora_f0[j - 1], mora_f0[j]
        if np.isnan(a_) or np.isnan(b_):
            continue
        tot += 1
        hit += (b_ > a_) if moras[j][2] == "H" else (b_ < a_)
    st = 12 * np.log2(np.exp(1)) * (np.nanpercentile(logf0, 95) - np.nanpercentile(logf0, 5))
    return dict(hit=int(hit), total=int(tot), f0_range_st=round(float(st), 1))
