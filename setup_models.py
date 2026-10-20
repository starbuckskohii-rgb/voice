"""Tải mọi thứ tts_ja.py cần, chạy một lần sau khi pip install -r requirements.txt.

- Từ điển UniDic đầy đủ cho fugashi (~500 MB)
- ASR SenseVoice int8 (~240 MB) để tự kiểm tra cách đọc -> asr_models/
- VoxCPM2 (~5 GB) -> models/VoxCPM2 (bỏ qua bằng --bo-voxcpm; khi đó tts_ja.py tự tải lúc chạy)
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASR_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
ASR_DIR = os.path.join(HERE, "asr_models", "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")


def unidic_ready() -> bool:
    try:
        import unidic

        return os.path.exists(os.path.join(unidic.DICDIR, "mecabrc"))
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bo-voxcpm", action="store_true", help="không tải VoxCPM2 về models/")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    if unidic_ready():
        print("UniDic: đã có")
    else:
        print("UniDic: đang tải...")
        subprocess.run([sys.executable, "-m", "unidic", "download"], check=True)

    for name in ("model.int8.onnx", "tokens.txt"):
        if os.path.exists(os.path.join(ASR_DIR, name)):
            print(f"ASR {name}: đã có")
        else:
            print(f"ASR {name}: đang tải...")
            hf_hub_download(ASR_REPO, name, local_dir=ASR_DIR)

    if not args.bo_voxcpm:
        target = os.path.join(HERE, "models", "VoxCPM2")
        if os.path.exists(os.path.join(target, "model.safetensors")):
            print("VoxCPM2: đã có")
        else:
            print("VoxCPM2: đang tải (~5 GB)...")
            snapshot_download("openbmb/VoxCPM2", local_dir=target)

    # pyopenjtalk tự tải từ điển open_jtalk (~20 MB) ở lần gọi đầu
    import pyopenjtalk

    print("OpenJTalk:", pyopenjtalk.g2p("二十歳の誕生日", kana=True))

    # thử nhanh phần tiền xử lý
    sys.path.insert(0, os.path.join(HERE, "code"))
    from ja_process_v2 import JapaneseTextPreprocessor

    print("Thử tiền xử lý:", JapaneseTextPreprocessor().process("庭には二羽鶏がいる。20歳の誕生日。"))
    print("Xong. Chạy thử:  python tts_ja.py \"ここでは靴を脱いでください\" -o ra.wav")


if __name__ == "__main__":
    main()
