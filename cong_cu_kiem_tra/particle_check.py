"""Kiểm tra tự động trợ từ は/へ/を trong audio có bị đọc thành "ha"/"he"/"wo" không.

Cách làm: cách đọc chuẩn (âm vị) lấy từ pyopenjtalk trên câu GỐC, đánh dấu vị trí
các trợ từ は/へ; nhận dạng âm vị audio bằng wav2vec2 (IPA); căn chỉnh hai chuỗi và
xem tại vị trí trợ từ có âm h (h, ɸ, ç, x) hay không.
"""

import difflib

import pyopenjtalk

from phoneme_rec import phonemes

_COARSE = {
    "ɯ": "u", "ʉ": "u", "ɨ": "u", "ʊ": "u", "ɪ": "i", "ɛ": "e", "ɔ": "o", "ɑ": "a", "æ": "a", "ʌ": "a", "ə": "a",
    "h": "h", "ɸ": "h", "ç": "h", "x": "h", "ɦ": "h",
    "w": "w", "v": "w", "β": "w", "ʋ": "w", "ɰ": "w",
}


def _coarse(tok: str) -> str:
    tok = tok.replace("ː", "").lower()
    if not tok:
        return ""
    if tok in _COARSE:
        return _COARSE[tok]
    if tok[0] in "aiueo":
        return tok[0]
    return _COARSE.get(tok[0], tok[0])


def expected_with_particles(sentence: str):
    """Chuỗi âm vị chuẩn (thô) + danh sách (vị trí bắt đầu, độ dài, chữ trợ từ)."""
    seq, particles = [], []
    for node in pyopenjtalk.run_frontend(sentence):
        pron = node["pron"].replace("’", "")
        if not pron or node["pos"] == "記号":
            continue
        phs = [p for p in pyopenjtalk.g2p(pron).split() if p not in ("pau", "cl")]
        phs = [_coarse(p) for p in phs]
        if node["pos"] == "助詞" and node["string"] in ("は", "へ", "を"):
            particles.append((len(seq), len(phs), node["string"]))
        seq.extend(phs)
    return seq, particles


def check(sentence: str, audio_path: str):
    exp, particles = expected_with_particles(sentence)
    rec_raw = phonemes(audio_path)
    rec = [c for c in (_coarse(t) for t in rec_raw.split()) if c]
    sm = difflib.SequenceMatcher(None, exp, rec, autojunk=False)
    # với mỗi vị trí trong exp: những âm vị nhận dạng được căn vào đó (kể cả chèn thêm ngay trước)
    aligned = {i: [] for i in range(len(exp) + 1)}
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                aligned[i1 + k].append(rec[j1 + k])
        elif op == "replace":
            for i in range(i1, i2):
                aligned[i].extend(rec[j1:j2])
        elif op == "insert":
            aligned[i1].extend(rec[j1:j2])
    results = []
    for start, length, ch in particles:
        got = [p for i in range(start, start + length) for p in aligned[i]]
        # は/へ sai = có âm h; を sai = có âm w ("wo")
        bad = "w" if ch == "を" else "h"
        results.append(dict(particle=ch, expected="".join(exp[start:start + length]), heard="".join(got),
                            misread=bad in got))
    return dict(phonemes=rec_raw, particles=results, ratio=round(sm.ratio(), 2))
