"""JapaneseTextPreprocessor v2: bản cập nhật pipeline trong ja_process.md cho VoxCPM2.

Thay đổi so với bản cũ (chi tiết và bằng chứng trong báo cáo):
- Layer 0: KHÔNG NFKC toàn bộ nữa (NFKC đổi ？→?, ！→!, ……→......). Chỉ đổi chữ/số
  full-width sang ASCII và katakana half-width sang full-width.
- Layer 1: từ điển ghi đè giờ dùng chung cơ chế với ja_particle_fix.READING_FIX
  (giữ chỗ, điền sau bước sửa trợ từ, để はたち không bị đổi thành わたち).
- Layer 2: KHÔNG đổi toàn câu sang hiragana nữa (gây mất ranh giới từ "none koto" và
  sai số đếm). Dùng fix_japanese_particles: sửa trợ từ は/へ/を theo cách phát âm,
  chỉ đổi số + từ đếm + ngày sang kana bằng pyopenjtalk, giữ nguyên kanji còn lại.
- Thêm: find_hard_reasons() để biết câu nào nên nhờ LLM ghi cách đọc {từ|cách đọc}.

Cài đặt:  pip install fugashi unidic pyopenjtalk && python -m unidic download
"""

import re
import unicodedata
from typing import Dict, List, Optional

import ja_particle_fix
from ja_hard_detect import find_hard_reasons
from ja_particle_fix import fix_japanese_particles


def _normalize_width(text: str) -> str:
    """Chữ/số full-width -> ASCII, katakana half-width -> full-width; giữ ？！……「」."""
    out = []
    for c in text:
        if ("０" <= c <= "９") or ("Ａ" <= c <= "Ｚ") or ("ａ" <= c <= "ｚ") or ("｡" <= c <= "ﾟ"):
            out.append(unicodedata.normalize("NFKC", c))
        else:
            out.append(c)
    # dấu ﾞﾟ half-width thành dấu kết hợp sau NFKC từng ký tự -> NFC ghép lại (ｶﾞ -> ガ)
    return unicodedata.normalize("NFC", "".join(out))


class JapaneseTextPreprocessor:
    def __init__(self, custom_dict: Optional[Dict[str, str]] = None, kana_mode: str = "units",
                 fix_particles: bool = False):
        # Mặc định = cấu hình tốt nhất đã kiểm chứng với Ultimate Cloning: giữ nguyên câu,
        # chỉ đổi ký hiệu đơn vị + cách đọc chỉ định. Không có prompt tiếng Nhật thì dùng
        # kana_mode="numbers", fix_particles=True.
        self.kana_mode = kana_mode
        self.fix_particles = fix_particles
        if custom_dict:
            ja_particle_fix.READING_FIX.update(custom_dict)

    def layer0_unicode_normalize(self, text: str) -> str:
        text = _normalize_width(text)
        text = text.replace("�", "")
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
        return text

    def hard_reasons(self, text: str) -> List:
        """Rỗng = câu dễ. Có lý do = nên nhờ LLM ghi cách đọc dạng {人気|ひとけ}."""
        return find_hard_reasons(self.layer0_unicode_normalize(text))

    def process(self, text: str) -> str:
        text = self.layer0_unicode_normalize(text)
        try:
            return fix_japanese_particles(text, kana_mode=self.kana_mode, fix_particles=self.fix_particles)
        except Exception as e:  # noqa: BLE001 - Layer 3: lỗi thì trả câu đã làm sạch
            print(f"[Japanese Preprocessor] Lỗi: {e}")
            return text


def trim_silence(wav, sr: int, threshold: float = 0.004, keep_head: float = 0.08, keep_tail: float = 0.3):
    """Cắt khoảng lặng đầu/cuối câu. VoxCPM2 hay để thừa 0.5-7 giây lặng ở cuối
    (78% file khi test), ghép nhiều câu sẽ thành chỗ ngắt dài bất thường. Ngưỡng thấp và
    giữ 0.3 giây đuôi để không mất âm "su" vô thanh cuối câu (〜ます)."""
    import numpy as np

    wav = np.asarray(wav)
    frame = max(1, int(sr * 0.02))
    n = len(wav) // frame
    if n == 0:
        return wav
    rms = np.sqrt((wav[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    voiced = np.where(rms > threshold)[0]
    if len(voiced) == 0:
        return wav
    start = max(0, voiced[0] * frame - int(sr * keep_head))
    end = min(len(wav), (voiced[-1] + 1) * frame + int(sr * keep_tail))
    return wav[start:end]


def join_sentences(wavs, sr: int, pause: float = 0.35):
    """Ghép các câu đã cắt lặng với khoảng nghỉ đều nhau (0.3-0.45 giây sau 。 là tự nhiên)."""
    import numpy as np

    gap = np.zeros(int(sr * pause), dtype=np.float32)
    out = []
    for i, w in enumerate(wavs):
        out.append(np.asarray(w, dtype=np.float32))
        if i < len(wavs) - 1:
            out.append(gap)
    return np.concatenate(out)


def synthesize(model, text: str, ref_wav: str, ref_text: str, pre: Optional[JapaneseTextPreprocessor] = None,
               trim: bool = True, **kwargs):
    """Cách gọi VoxCPM2 cho tiếng Nhật đã kiểm chứng tốt nhất: Ultimate Cloning.

    ref_wav: giọng mẫu tiếng Nhật 5-10 giây; ref_text: lời của giọng mẫu (chữ gốc).
    Cả lời giọng mẫu lẫn câu cần đọc đều qua cùng bộ tiền xử lý, để model thấy
    prompt và câu mới cùng một kiểu chữ. Không có prompt tiếng Nhật, VoxCPM2 hay
    đọc câu bắt đầu bằng kanji (母, 十分, 「本当) bằng tiếng Trung hoặc méo chữ đầu.
    """
    pre = pre or JapaneseTextPreprocessor()
    wav = model.generate(
        text=pre.process(text),
        prompt_wav_path=ref_wav,
        prompt_text=pre.process(ref_text),
        reference_wav_path=ref_wav,
        normalize=False,  # normalize=True coi kanji là tiếng Trung (text_normalize.py)
        **kwargs,
    )
    return trim_silence(wav, model.tts_model.sample_rate) if trim else wav


def _reading(text: str) -> str:
    import pyopenjtalk

    text = re.sub(r"\{([^{}|]*)\|[^{}|]*\}", r"\1", text)  # {人気|ひとけ} -> 人気 (so theo chữ gốc)
    kana = pyopenjtalk.g2p(text, kana=True) if text.strip() else ""
    return re.sub(r"[、。！？「」…\s,.!?%％・ー\-−]", "", kana)


def check_reading(text: str, heard: str):
    """(độ khớp cách đọc 0-1, số mora thừa ở cuối) giữa câu gốc và chữ ASR nghe được."""
    import difflib

    ref, hyp = _reading(text), _reading(heard)
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    extra = sum(j2 - j1 for op, i1, i2, j1, j2 in sm.get_opcodes() if op == "insert" and i1 == len(ref))
    return sm.ratio(), extra


def missing_counters(text: str, heard: str):
    """Các cụm số + từ đếm (三階 サンガイ, 一回 イッカイ, 4月1日 シガツツイタチ) mà ASR không nghe
    ra đúng. Sai 1 mora (三階 sangai -> sankai) chỉ hạ độ khớp cả câu xuống ~0.97 nên
    phải kiểm tra riêng từng cụm."""
    import pyopenjtalk

    from ja_hard_detect import _normalize_pron
    from ja_particle_fix import _is_number_node, _node_reading

    text = re.sub(r"\{([^{}|]*)\|[^{}|]*\}", r"\1", text)
    groups, cur = [], []
    for node in pyopenjtalk.run_frontend(text):
        if _is_number_node(node):
            cur.append(node)
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    heard_reading = _normalize_pron(pyopenjtalk.g2p(heard, kana=True)) if heard.strip() else ""
    missing = []
    for g in groups:
        want = _normalize_pron("".join(_node_reading(n) for n in g))
        if want and want not in heard_reading:
            missing.append("".join(n["string"] for n in g))
    return missing


def make_sensevoice_transcriber(model_dir: str):
    """ASR kiểm tra nhanh bằng SenseVoice (sherpa-onnx). model_dir = thư mục
    sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 (tải từ GitHub releases của k2-fsa/sherpa-onnx)."""
    import numpy as np
    import sherpa_onnx
    from scipy.signal import resample_poly

    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{model_dir}/model.int8.onnx", tokens=f"{model_dir}/tokens.txt", language="ja", use_itn=False)

    def transcribe(wav, sr):
        x = resample_poly(np.asarray(wav, dtype=np.float32), 16000, sr).astype(np.float32)
        x = np.concatenate([np.zeros(8000, np.float32), x, np.zeros(4800, np.float32)])  # ASR hụt âm đầu nếu không đệm
        s = rec.create_stream()
        s.accept_waveform(16000, x)
        rec.decode_stream(s)
        return s.result.text

    return transcribe


def unexpected_pauses(wav, sr: int, text: str, min_pause: float = 0.2, rel_threshold: float = 0.1):
    """Số chỗ ngắt (lặng >= min_pause giây) giữa câu vượt quá số dấu ngắt trong chữ
    (、。！？…). VoxCPM2 thỉnh thoảng ngắt giữa cụm (三匹の… 猫と, hoặc ね… こと khi
    đầu vào là hiragana), nghe như ngập ngừng."""
    import numpy as np

    x = np.asarray(wav, dtype=np.float32)
    frame = int(sr * 0.01)
    n = len(x) // frame
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    thr = rel_threshold * np.sqrt(np.mean(x ** 2))
    voiced = np.where(rms > thr)[0]
    if len(voiced) == 0:
        return 0
    inner = rms[voiced[0]: voiced[-1] + 1] <= thr
    pauses, run = 0, 0
    for silent in inner:
        run = run + 1 if silent else 0
        if run == int(min_pause / 0.01):
            pauses += 1
    body = re.sub(r"[。！？!?…]+$", "", text.strip())
    allowed = len(re.findall(r"[、，,。！？!?…]+", body))
    return max(0, pauses - allowed)


def synthesize_checked(model, text: str, ref_wav: str, ref_text: str, transcribe, pre=None, seed: int = 0,
                       max_tries: int = 3, min_match: float = 0.85, max_extra: int = 2, **kwargs):
    """Như synthesize(), nhưng nghe lại bằng ASR rồi sinh lại với seed khác khi: đọc
    thiếu/sai (khớp < min_match), đọc thừa ở cuối, sai cụm số/từ đếm (missing_counters),
    hoặc ngắt nhịp giữa cụm (unexpected_pauses). Trả (wav, số lần thử, độ khớp)."""
    sr = model.tts_model.sample_rate
    best = None
    for attempt in range(max_tries):
        wav = synthesize(model, text, ref_wav, ref_text, pre=pre, seed=seed + attempt, **kwargs)
        heard = transcribe(wav, sr)
        match, extra = check_reading(text, heard)
        processed = (pre or JapaneseTextPreprocessor()).process(text)
        if missing_counters(processed, heard) or unexpected_pauses(wav, sr, processed):
            match = min(match, min_match - 0.01)  # sai cụm số/từ đếm, hoặc ngắt nhịp giữa cụm: sinh lại
        if match >= min_match and extra <= max_extra:
            return wav, attempt + 1, match
        if best is None or match - 0.05 * extra > best[2] - 0.05 * best[3]:
            best = (wav, attempt + 1, match, extra)
    return best[0], max_tries, best[2]


if __name__ == "__main__":
    p = JapaneseTextPreprocessor()
    for s in ["私は学校へ行きます。本を読む。", "8月の誕生日に、三階で三匹の猫と一回だけ遊んだ。",
              "「本当ですか？」えっと……ｶﾞｯｺｳは１０時から！", "人気のない場所で待っていた。"]:
        print(s, "\n  ->", p.process(s), "\n  khó:", p.hard_reasons(s))
