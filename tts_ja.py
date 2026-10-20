"""Đọc tiếng Nhật bằng VoxCPM2: tiền xử lý chữ + Ultimate Cloning + tự kiểm tra bằng ASR.

Ví dụ:
  python tts_ja.py "母は部屋へ入って、水を飲んでから東京へ出かけた。" -o ra.wav
  python tts_ja.py "ここでは靴を脱いでください" --giong nu -o ra.wav
  python tts_ja.py -f examples/cau_kho_test.txt -o ra/           # mỗi dòng một file
  python tts_ja.py -f examples/doan_van.txt --noi -o doan_van.wav  # cả đoạn thành một file
  python tts_ja.py "..." --voice giong_cua_toi.wav --voice-text "lời đúng từng chữ của file mẫu"

Chạy setup_models.py một lần trước (tải model ASR, từ điển UniDic, VoxCPM2).
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))

ASR_DIR = os.path.join(HERE, "asr_models", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
LOCAL_MODEL = os.path.join(HERE, "models", "VoxCPM2")


def split_sentences(text: str):
    """Tách đoạn văn thành câu sau 。！？ (không tách bên trong 「」『』)."""
    out, cur, depth = [], "", 0
    for ch in text:
        cur += ch
        if ch in "「『（(":
            depth += 1
        elif ch in "」』）)":
            depth = max(0, depth - 1)
        elif ch == "\n" or (ch in "。！？!?" and depth == 0):
            if cur.strip():
                out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    # gộp đuôi kiểu 」と言った。 hay ？！ dính liền vào câu trước
    merged = []
    for s in out:
        if merged and re.fullmatch(r"[。！？!?」』）)]+", s):
            merged[-1] += s
        else:
            merged.append(s)
    return merged


def load_voice(args):
    if args.voice:
        if not args.voice_text:
            sys.exit("Thiếu --voice-text: lời của file giọng mẫu, đúng từng chữ (sai lời -> đầu câu lẫn âm lạ).")
        return args.voice, args.voice_text
    wav = os.path.join(HERE, "voices", f"{args.giong}.wav")
    with open(os.path.join(HERE, "voices", f"{args.giong}.txt"), encoding="utf-8") as f:
        return wav, f.read().strip()


def main():
    ap = argparse.ArgumentParser(description="Đọc tiếng Nhật bằng VoxCPM2 (tiền xử lý + tự kiểm tra).")
    ap.add_argument("text", nargs="?", help="câu/đoạn cần đọc")
    ap.add_argument("-f", "--file", help="file .txt, mỗi dòng một câu")
    ap.add_argument("-o", "--out", default=None, help="file .wav/.mp3, hoặc thư mục (khi đọc nhiều câu)")
    ap.add_argument("--noi", action="store_true", help="ghép tất cả câu thành một file")
    ap.add_argument("--giong", default="nam", help="giọng mẫu có sẵn trong voices/ (nam, nu)")
    ap.add_argument("--voice", help="file giọng mẫu riêng (5-10 giây, người thật, ít nhiễu)")
    ap.add_argument("--voice-text", help="lời của file giọng mẫu, đúng từng chữ")
    ap.add_argument("--dict", default=os.path.join(HERE, "tu_dien.json"), help="từ điển cách đọc {từ: cách đọc}")
    ap.add_argument("--model", default=None, help="thư mục model hoặc id Hugging Face (mặc định openbmb/VoxCPM2)")
    ap.add_argument("--device", default=None, help="cuda / cpu (mặc định tự chọn)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tries", type=int, default=5, help="số lần sinh lại tối đa khi kiểm tra thấy lỗi")
    ap.add_argument("--khong-kiem-tra", action="store_true", help="bỏ bước nghe lại bằng ASR (nhanh hơn, dễ lọt lỗi)")
    ap.add_argument("--nghi", type=float, default=0.35, help="giây nghỉ giữa các câu khi --noi")
    args = ap.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            sentences = [s for line in f for s in split_sentences(line)]
    elif args.text:
        sentences = split_sentences(args.text)
    else:
        ap.error("cần câu cần đọc hoặc -f file.txt")
    if not sentences:
        ap.error("không có câu nào để đọc")
    one_file = args.noi or len(sentences) == 1
    out = args.out or ("ra.wav" if one_file else "ra")

    import numpy as np
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    from ja_process_v2 import (JapaneseTextPreprocessor, join_sentences, make_sensevoice_transcriber,
                               reading_problems, synthesize, synthesize_checked)

    custom = {}
    if args.dict and os.path.exists(args.dict):
        with open(args.dict, encoding="utf-8") as f:
            custom = json.load(f)
    pre = JapaneseTextPreprocessor(custom_dict=custom)
    ref_wav, ref_text = load_voice(args)

    transcribe = None
    if not args.khong_kiem_tra:
        if os.path.exists(os.path.join(ASR_DIR, "model.int8.onnx")):
            transcribe = make_sensevoice_transcriber(ASR_DIR)
        else:
            print("! Chưa có model ASR (chạy setup_models.py) -> đọc không tự kiểm tra.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_id = args.model or (LOCAL_MODEL if os.path.exists(os.path.join(LOCAL_MODEL, "config.json")) else "openbmb/VoxCPM2")
    print(f"Nạp {model_id} trên {device}...")
    model = VoxCPM.from_pretrained(model_id, load_denoiser=False, optimize=device == "cuda", device=device)
    sr = model.tts_model.sample_rate

    wavs = []
    for i, s in enumerate(sentences, 1):
        hard = pre.hard_reasons(pre.process(s))  # sau khi áp từ điển: từ đã có cách đọc thì không báo
        if hard:
            print(f"  ! câu khó {hard}: nếu đọc sai, ghi cách đọc {{từ|cách đọc}} hoặc thêm vào {os.path.basename(args.dict)}")
        t0 = time.time()
        if transcribe:
            wav, tries, match = synthesize_checked(model, s, ref_wav, ref_text, transcribe, pre=pre,
                                                   seed=args.seed, max_tries=args.max_tries)
            heard = transcribe(wav, sr)
            problems = reading_problems(wav, sr, s, heard, pre)[0]
            note = f"{tries} lần, khớp {match:.2f}" + (f" | CHƯA ĐẠT {problems}, nên nghe lại" if problems else "")
            note += f" | ASR nghe: {heard}"
        else:
            wav = synthesize(model, s, ref_wav, ref_text, pre=pre, seed=args.seed)
            note = "không kiểm tra"
        print(f"[{i}/{len(sentences)}] {s} -> {pre.process(s)} | {time.time() - t0:.0f}s, {note}", flush=True)
        wavs.append(np.asarray(wav, dtype=np.float32))

    if one_file:
        audio = join_sentences(wavs, sr, pause=args.nghi) if len(wavs) > 1 else wavs[0]
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        sf.write(out, audio, sr)
        print(f"Đã ghi {out}")
    else:
        out = os.path.normpath(out)
        os.makedirs(out, exist_ok=True)
        for i, w in enumerate(wavs, 1):
            sf.write(os.path.join(out, f"{i:03d}.wav"), w, sr)
        with open(os.path.join(out, "danh_sach.txt"), "w", encoding="utf-8") as f:
            f.write("".join(f"{i:03d}.wav\t{s}\n" for i, s in enumerate(sentences, 1)))
        print(f"Đã ghi {len(wavs)} file vào {out}/ (danh_sach.txt ghi câu của từng file)")


if __name__ == "__main__":
    main()
