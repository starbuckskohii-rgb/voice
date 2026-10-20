# Tiền xử lý tiếng Nhật cho VoxCPM2

## Dùng ngay
```
pip install -r requirements.txt && python -m unidic download
```
```python
from voxcpm import VoxCPM
from ja_process_v2 import (JapaneseTextPreprocessor, synthesize, synthesize_checked,
                           make_sensevoice_transcriber, join_sentences)

model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
pre = JapaneseTextPreprocessor(custom_dict={"VoxCPM": "ボックスシーピーエム"})

text = "人気のない場所で、彼は一人で待っていた。"
if pre.hard_reasons(text):          # câu khó -> nhờ LLM ghi cách đọc {人気|ひとけ}
    text = rewrite_with_llm(text)
# Tự kiểm tra: đọc thiếu/thừa/sai (ASR), sai cụm số + từ đếm, ngắt nhịp giữa cụm
# (khoảng lặng ≥0,2 s ở chỗ không có dấu câu) -> sinh lại seed khác (khuyên dùng):
tr = make_sensevoice_transcriber("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
wav, tries, match = synthesize_checked(model, text, "giong_mau.wav", "lời giọng mẫu, đúng từng chữ", tr)
# hoặc không kiểm tra: wav = synthesize(model, text, ref_wav=..., ref_text=...)
```
Giọng mẫu quyết định độ tự nhiên: dùng bản thu người thật, rõ, ít nhiễu, 5-10 giây
(giọng mẫu UTMOS 2.1 -> đầu ra 1.8; giọng người thật UTMOS 3.9 -> đầu ra 3.8, ngang người).
Giọng mẫu tự tạo cũng hay làm AI ngắt nhịp giữa cụm (28% câu với giọng nữ mẫu, 58% với giọng nam mẫu;
giọng người thật clone 8%, người thật đọc 2,5%).

## Thư mục
- code/ja_process_v2.py — thay cho class JapaneseTextPreprocessor cũ; có hàm synthesize() gọi Ultimate Cloning
- code/ja_particle_fix.py — bảng đơn vị (km, kg, cm, ℃), READING_FIX, cú pháp {từ|cách đọc}; chế độ đổi trợ từ/số
  (kana_mode="numbers", fix_particles=True) chỉ dùng khi KHÔNG có prompt tiếng Nhật
- code/ja_hard_detect.py — phát hiện câu khó để gửi LLM
- ref_ja_female.wav / ref_ja_male.wav — giọng mẫu dùng khi test (VoxCPM2 tự tạo bằng Voice Design)
  lời: こんにちは。今日は、私たちの町にできた新しい図書館について、少しお話ししたいと思います。
       はじめまして。私は、この町で二十年ほどパン屋を営んでいます。今日は、その思い出をお話しします。
- cong_cu_kiem_tra — sinh audio hàng loạt (fix_test.py); chấm phát âm bằng ASR + âm vị (eval_fix.py);
  chấm trọng âm + độ tự nhiên UTMOS (prosody_check.py, naturalness_eval.py, paired_eval.py).
  Model (VoxCPM2, ASR sherpa-onnx, wav2vec2 âm vị, UTMOS) tải riêng vào models/ và asr_models/.
  Biến thể U của ab_test.py cần pipeline cũ đặt ở cong_cu_kiem_tra/user_pipeline/user_preproc.py (không kèm trong repo).
