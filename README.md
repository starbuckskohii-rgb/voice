# Đọc tiếng Nhật bằng VoxCPM2

Tiền xử lý chữ + cách gọi VoxCPM2 đã kiểm chứng (Ultimate Cloning) + tự nghe lại bằng ASR,
sai thì sinh lại. Xử lý các lỗi hay gặp: trợ từ は/へ/を đọc thành ha/he/wo, đầu câu lẫn âm lạ
hoặc giọng Trung, sai số đếm (三階 sankai), ngắt nhịp giữa từ (ね…こと), đọc thừa/lảm nhảm,
đơn vị (km, kg, ℃), từ nhiều cách đọc (二十歳, 鶏).

## Cài đặt (một lần)

Cần Python 3.10+ và git. Có GPU NVIDIA thì cài torch bản CUDA trước theo
https://pytorch.org/get-started/locally/ (không có GPU vẫn chạy được, chậm hơn).

```
git clone https://github.com/starbuckskohii-rgb/voice.git
cd voice
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python setup_models.py             # tải UniDic, ASR kiểm tra (~240 MB), VoxCPM2 (~5 GB)
```

## Dùng

```
python tts_ja.py "母は部屋へ入って、水を飲んでから東京へ出かけた。" -o ra.wav
python tts_ja.py "ここでは靴を脱いでください" --giong nu -o ra.wav
python tts_ja.py -f examples/cau_kho_test.txt -o ra/              # mỗi dòng một file
python tts_ja.py -f examples/doan_van.txt --noi -o doan_van.wav   # cả đoạn thành một file
```

Mỗi câu in ra: câu sau tiền xử lý, số lần sinh, chữ ASR nghe được. Câu nào hết lượt
(`--max-tries`, mặc định 5) vẫn còn lỗi thì in `CHƯA ĐẠT [...]` để bạn nghe lại.

### Giọng của bạn

```
python tts_ja.py "..." --voice giong_cua_toi.wav --voice-text "lời đúng từng chữ của file mẫu"
```

- File mẫu: người thật, 5-10 giây, rõ, không nhạc nền. Giọng mẫu quyết định độ tự nhiên
  (mẫu UTMOS 2.1 -> đầu ra 1.8; mẫu người thật 3.9 -> đầu ra 3.8, ngang người đọc).
- `--voice-text` phải khớp **đúng từng chữ** với audio mẫu. Thiếu chữ -> model đọc tiếp phần
  lời bị thiếu, đầu câu lẫn âm lạ và lảm nhảm cuối câu. Muốn dùng lâu dài: chép file vào
  `voices/ten.wav` + lời vào `voices/ten.txt`, rồi gọi `--giong ten`.
- `voices/nam.wav`, `voices/nu.wav` là giọng VoxCPM2 tự tạo (Voice Design), dùng để test.

### Từ đọc sai: từ điển và cú pháp {từ|cách đọc}

- `tu_dien.json`: `{"từ": "cách đọc"}` áp cho mọi câu (chỉ khớp nguyên từ, ví dụ 鶏 không đụng 鶏肉).
  Dùng katakana cho cách đọc nếu có thể.
- Trong câu: `{人気|ひとけ}のない場所で` để chỉ định cách đọc cho riêng chỗ đó.
- Câu có từ nhiều cách đọc (人気, 市場, 生物, 一日...) được in cảnh báo `câu khó`.
  Biết là pyopenjtalk đọc sai nhưng chưa được cảnh báo: 方 (かた/ほう), 辛い (からい/つらい).

### Gọi từ code Python

```python
import sys; sys.path.insert(0, "code")
from voxcpm import VoxCPM
from ja_process_v2 import JapaneseTextPreprocessor, synthesize_checked, make_sensevoice_transcriber

model = VoxCPM.from_pretrained("models/VoxCPM2", load_denoiser=False)
pre = JapaneseTextPreprocessor(custom_dict={"VoxCPM": "ボックスシーピーエム"})
tr = make_sensevoice_transcriber("asr_models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
wav, tries, match = synthesize_checked(model, "庭には二羽鶏がいる。", "voices/nam.wav",
                                       open("voices/nam.txt", encoding="utf-8").read().strip(), tr, pre=pre,
                                       max_tries=5)
```

## Cách hoạt động

1. **Tiền xử lý** (`code/ja_process_v2.py`, `code/ja_particle_fix.py`): giữ nguyên kanji và trợ
   từ (đổi は→わ hay đổi cả câu sang hiragana làm model mất ranh giới từ: ここでわ靴 -> 和靴,
   猫と -> ね…こと). Chỉ đổi đơn vị (km -> キロメートル), chữ full-width, và các cách đọc chỉ định
   (二十歳/20歳 -> ハタチ, 鶏 -> ニワトリ: để kanji thì 二羽鶏 bị nuốt mất một にわ).
2. **Gọi VoxCPM2 kiểu Ultimate Cloning**: `prompt_wav` + `prompt_text` + `reference_wav`,
   `normalize=False` (normalize=True coi kanji là tiếng Trung). Không có prompt tiếng Nhật thì
   đầu câu hay lẫn âm lạ/giọng Trung, nuốt chữ.
3. **Tự kiểm tra** (`reading_problems`): ASR SenseVoice nghe lại, loại bản có
   - độ khớp cách đọc cả câu < 0.85, hoặc đọc thừa > 2 mora ở đầu/cuối câu,
   - cụm số + từ đếm sai (三階 サンガイ), từ có kanji sai (祝った -> いった),
   - khoảng lặng ≥ 0.15 s bên trong cụm (giữa từ, sau の, trước trợ từ, giữa các phần của số);
     định vị bằng mốc thời gian ASR, không tính dấu câu, âm ッ, ranh giới cụm sau は/から.
   Lỗi thì sinh lại seed khác; hết lượt thì lấy bản ít lỗi nhất.
   Đo trên 40 câu người thật đọc: không loại nhầm vì ngắt nhịp/đọc thừa; kiểm tra từng từ
   loại nhầm 5/40 do ASR nghe sai từ hiếm (木曽川) -> chỉ tốn thêm một lần sinh.

## Thư mục

- `tts_ja.py` — lệnh đọc; `setup_models.py` — tải model
- `code/` — thư viện: `ja_process_v2.py` (tiền xử lý, gọi model, kiểm tra), `ja_particle_fix.py`
  (đơn vị, READING_FIX, {từ|cách đọc}), `ja_hard_detect.py` (phát hiện câu khó)
- `voices/` — giọng mẫu (`.wav`) + lời (`.txt`); `tu_dien.json` — từ điển cách đọc
- `examples/` — 60 câu khó để test (`cau_kho_test.tsv` có cách đọc đúng + chỗ cần nghe), đoạn văn mẫu
- `cong_cu_kiem_tra/` — công cụ đánh giá đã dùng khi tinh chỉnh (chấm phát âm bằng ASR + âm vị,
  trọng âm, UTMOS, hiệu chỉnh ngưỡng `pause_calib.py`, `word_calib.py`); cần `requirements-eval.txt`
  và các model đánh giá riêng. Biến thể U của `ab_test.py` cần pipeline cũ ở
  `cong_cu_kiem_tra/user_pipeline/user_preproc.py` (không kèm).

## Lưu ý

- CPU: mỗi câu ngắn mất ~15-60 giây mỗi lần sinh; GPU nhanh hơn nhiều.
- Windows: nếu `pip install pyopenjtalk` lỗi biên dịch, cài `pyopenjtalk-prebuilt` thay thế
  (chưa test trên Windows).
- Kết quả kiểm tra là đo bằng máy (ASR); vẫn nên nghe lại câu quan trọng.
