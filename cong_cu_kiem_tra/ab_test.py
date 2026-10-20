"""Sinh audio VoxCPM2 cho bộ câu ja_corpus theo nhiều cách tiền xử lý (A/B test).

Chạy:  python ab_test.py --plan main --out ab_out
Chạy lại được: file .wav đã có thì bỏ qua. Mọi job ghi vào ab_out/manifest.jsonl.
"""

import argparse
import json
import os
import sys
import time

import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "user_pipeline"))

from ja_corpus import CORPUS  # noqa: E402
from ja_particle_fix import fix_japanese_particles  # noqa: E402


def user_pipeline(text):
    """Pipeline hiện tại của bạn (ja_process.md), dùng unidic-lite như tài liệu cài."""
    import fugashi
    import unidic_lite
    from user_preproc import JapaneseTextPreprocessor

    p = JapaneseTextPreprocessor()
    p.tagger = fugashi.Tagger("-d " + unidic_lite.DICDIR)
    return p.process(text)


VARIANTS = {
    "U": user_pipeline,                                                   # pipeline hiện tại của bạn
    "R": lambda t: t,                                                     # câu gốc, không xử lý
    "A": lambda t: fix_japanese_particles(t),                             # đề xuất
    "Y": lambda t: fix_japanese_particles(t, fix_particles=False),        # A nhưng giữ は/へ/を
    "K": lambda t: fix_japanese_particles(t, number_script="katakana"),   # A, cụm số viết katakana
    "B": lambda t: fix_japanese_particles(t, bunsetsu_spaces=True),       # A + khoảng trắng bunsetsu
    "W": lambda t: fix_japanese_particles(t, convert_wo=False),           # A nhưng giữ を
}

ALL_IDS = [c[0] for c in CORPUS]
USER_IDS = ["p1", "p2", "n1", "n2", "n3", "t1", "u2", "q1"]          # minh hoạ lỗi pipeline cũ
PARTICLE_IDS = ["p1", "p2", "p3", "p4", "q2", "k1", "k2", "s1"]       # câu nhiều は/へ/を
NUMBER_IDS = ["n1", "n3", "t1", "t2", "t3", "u1", "u2", "u3"]

PLANS = {
    # (danh sách câu, biến thể, seeds) — rút gọn để chạy CPU trong ~1 giờ
    "main": [(ALL_IDS, "A", [0]), (USER_IDS, "U", [0])],
    "particles": [(PARTICLE_IDS, "A", [1]), (PARTICLE_IDS, "Y", [0, 1])],
    "numbers": [(NUMBER_IDS, "R", [0]), (["t1", "u1", "u2", "u3"], "K", [0])],
    "bunsetsu": [(["n1", "p2", "s1", "t1"], "B", [0])],
    "wo": [(["p2", "p4", "n4", "k2", "s1", "l1"], "W", [0, 1]), (["n4", "l1"], "A", [1])],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(HERE, "models/VoxCPM2"))
    ap.add_argument("--out", default=os.path.join(HERE, "ab_out"))
    ap.add_argument("--plan", nargs="+", default=["main"])
    ap.add_argument("--reference", default=None, help="file giọng mẫu (tuỳ chọn)")
    args = ap.parse_args()

    import torch
    from voxcpm import VoxCPM

    torch.set_num_threads(os.cpu_count())
    os.makedirs(args.out, exist_ok=True)
    corpus = {c[0]: c for c in CORPUS}

    jobs = [(sid, v, seed) for plan in args.plan for ids, v, seeds in PLANS[plan] for sid in ids for seed in seeds]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(args.out, f"{j[0]}_{j[1]}_seed{j[2]}.wav"))]
    print(f"{len(jobs)} job cần chạy", flush=True)
    if not jobs:
        return

    t0 = time.time()
    model = VoxCPM.from_pretrained(args.model, load_denoiser=False, optimize=False, device="cpu")
    sr = model.tts_model.sample_rate
    print(f"load model: {time.time() - t0:.1f}s", flush=True)

    for i, (sid, v, seed) in enumerate(jobs, 1):
        _, group, sentence, expected = corpus[sid]
        text_in = VARIANTS[v](sentence)
        path = os.path.join(args.out, f"{sid}_{v}_seed{seed}.wav")
        kwargs = dict(text=text_in, normalize=False, seed=seed)
        if args.reference:
            kwargs["reference_wav_path"] = args.reference
        t0 = time.time()
        wav = model.generate(**kwargs)
        sf.write(path + ".tmp.wav", wav, sr)
        os.replace(path + ".tmp.wav", path)
        row = dict(file=path, id=sid, group=group, variant=v, seed=seed, sentence=sentence,
                   expected=expected, text_in=text_in, audio_sec=round(len(wav) / sr, 2),
                   gen_sec=round(time.time() - t0, 1))
        with open(os.path.join(args.out, "manifest.jsonl"), "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(jobs)}] {sid}_{v}_seed{seed} {row['audio_sec']}s audio / {row['gen_sec']}s | {text_in}", flush=True)


if __name__ == "__main__":
    main()
