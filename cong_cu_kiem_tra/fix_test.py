"""Chạy các job thử nghiệm sửa lỗi cho VoxCPM2 từ file JSON.

Mỗi job: {"name": ..., "text": ..., "seed": 0, "ref": "đường dẫn giọng mẫu" (tuỳ chọn),
          "prompt_wav": ..., "prompt_text": ... (tuỳ chọn, Ultimate Cloning)}
Chạy:  python fix_test.py jobs.json fix_out
Bỏ qua job đã có file .wav. Ghi fix_out/manifest.jsonl.
"""

import json
import os
import sys
import time

import soundfile as sf
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    jobs_path, out = sys.argv[1], sys.argv[2]
    jobs = json.load(open(jobs_path))
    os.makedirs(out, exist_ok=True)
    jobs = [j for j in jobs if not os.path.exists(os.path.join(out, j["name"] + ".wav"))]
    print(f"{len(jobs)} job", flush=True)
    if not jobs:
        return
    from voxcpm import VoxCPM

    torch.set_num_threads(os.cpu_count())
    model = VoxCPM.from_pretrained(os.path.join(HERE, "models/VoxCPM2"), load_denoiser=False, optimize=False, device="cpu")
    sr = model.tts_model.sample_rate
    for i, j in enumerate(jobs, 1):
        kwargs = dict(text=j["text"], normalize=False, seed=j.get("seed", 0))
        if j.get("ref"):
            kwargs["reference_wav_path"] = j["ref"]
        if j.get("prompt_wav"):
            kwargs["prompt_wav_path"] = j["prompt_wav"]
            kwargs["prompt_text"] = j["prompt_text"]
        t0 = time.time()
        wav = model.generate(**kwargs)
        path = os.path.join(out, j["name"] + ".wav")
        sf.write(path + ".tmp.wav", wav, sr)
        os.replace(path + ".tmp.wav", path)
        row = dict(j, file=path, audio_sec=round(len(wav) / sr, 2), gen_sec=round(time.time() - t0, 1))
        with open(os.path.join(out, "manifest.jsonl"), "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(jobs)}] {j['name']} {row['audio_sec']}s / {row['gen_sec']}s | {j['text']}", flush=True)


if __name__ == "__main__":
    main()
