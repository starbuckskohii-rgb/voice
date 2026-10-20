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
    """(độ khớp cách đọc 0-1, số mora thừa ở đầu + cuối câu) giữa câu gốc và chữ ASR nghe
    được. Thừa ở đầu: model lẫn âm lạ trước câu (昼下がり一人で…); thừa ở cuối: lảm nhảm."""
    import difflib

    ref, hyp = _reading(text), _reading(heard)
    sm = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    if not blocks:
        return sm.ratio(), len(hyp)
    first, last = blocks[0], blocks[-1]
    lead = max(0, first.b - first.a)
    trail = max(0, (len(hyp) - last.b - last.size) - (len(ref) - last.a - last.size))
    return sm.ratio(), lead + trail


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


def missing_words(text: str, heard: str):
    """Các từ có kanji (danh, động, tính, phó từ) mà ASR không nghe ra đúng cách đọc.
    Sai một từ (祝った iwatta -> itta) chỉ hạ độ khớp cả câu xuống ~0.86 nên phải kiểm
    tra riêng; cụm số/từ đếm do missing_counters lo."""
    import pyopenjtalk

    from ja_hard_detect import _normalize_pron
    from ja_particle_fix import _is_number_node, _node_reading

    text = re.sub(r"\{([^{}|]*)\|[^{}|]*\}", r"\1", text)
    heard_reading = _normalize_pron(pyopenjtalk.g2p(heard, kana=True)) if heard.strip() else ""
    missing, prev = [], ""
    for n in pyopenjtalk.run_frontend(text):
        reading = _node_reading(n)
        if (n["pos"] in ("名詞", "動詞", "形容詞", "副詞") and not _is_number_node(n)
                and re.search(r"[一-鿿々]", n["string"])):
            want = _normalize_pron(reading)
            # chuẩn hoá trường âm phụ thuộc âm đứng trước (で+イキ -> デエキ): so cả dạng có ngữ cảnh
            in_ctx = _normalize_pron(prev[-1:] + reading)[1:] if prev else want
            if len(want) >= 2 and want not in heard_reading and in_ctx not in heard_reading:
                missing.append(n["string"])
        prev = _normalize_pron(reading) or prev
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
        r = s.result
        return Heard(r.text, [(tok, t - 0.5) for tok, t in zip(r.tokens, r.timestamps)])

    return transcribe


class Heard(str):
    """Chữ ASR nghe được, kèm mốc thời gian từng token [(token, giây)] để định vị chỗ ngắt."""

    def __new__(cls, text, times=None):
        obj = super().__new__(cls, text)
        obj.times = times or []
        return obj


def _silences(wav, sr: int, min_pause: float, rel_threshold: float):
    """Các khoảng lặng giữa câu [(bắt đầu, kết thúc) giây] dài >= min_pause."""
    import numpy as np

    x = np.asarray(wav, dtype=np.float32)
    frame = int(sr * 0.01)
    n = len(x) // frame
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    thr = rel_threshold * np.sqrt(np.mean(x ** 2))
    voiced = np.where(rms > thr)[0]
    if len(voiced) == 0:
        return []
    out, start = [], None
    for i in range(voiced[0], voiced[-1] + 2):
        silent = i <= voiced[-1] and rms[i] <= thr
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            if i - start >= round(min_pause / 0.01):
                out.append((start / 100, i / 100))
            start = None
    return out


def _pron_nodes(text: str):
    """Các từ theo pyopenjtalk, thêm khoá "r" = cách đọc chuẩn hoá (bỏ dấu trọng âm, ー)."""
    import pyopenjtalk

    nodes = pyopenjtalk.run_frontend(text)
    for n in nodes:
        n["r"] = re.sub(r"[’'ー]", "", n["pron"])
    return nodes


def _tight_boundary(prev, nxt) -> bool:
    """Ranh giới bên trong một cụm, người Nhật không ngắt ở đây: trước trợ từ/trợ động
    từ/hậu tố (猫|と), sau の bổ nghĩa (三匹の|猫), sau tiền tố/連体詞 (この|本), giữa các
    phần của một con số (2026年|10月), giữa hai danh từ ghép (最高|気温)."""
    from ja_particle_fix import _is_number_node

    if nxt["pos"] in ("助詞", "助動詞") or nxt["pos_group1"] == "接尾":
        return True
    if prev["pos"] in ("接頭詞", "連体詞") or (prev["pos"] == "助詞" and prev["pos_group1"] == "連体化"):
        return True
    if _is_number_node(prev) and _is_number_node(nxt):
        return True
    return prev["pos"] == "名詞" and nxt["pos"] == "名詞" and prev["pos_group1"] != "副詞可能"


def _pause_allowed(heard: str, cut: int, text: str) -> bool:
    """Khoảng lặng ngay sau heard[:cut] có hợp lệ không. Hợp lệ: ở đầu/cuối câu, ở âm ッ
    (closure tự nhiên), đúng chỗ có dấu câu trong chữ gốc, hoặc ở ranh giới cụm (sau は,
    から, と...). Không hợp lệ: giữa một từ (ね|こ) hoặc bên trong một cụm (_tight_boundary)."""
    import difflib

    if cut <= 0 or cut >= len(heard):
        return True
    # pyopenjtalk đổi số sang kanji (65 -> 六十五) nên tính lại vị trí cắt trên chữ đã đổi
    cut = len("".join(n["string"] for n in _pron_nodes(heard[:cut])))
    nodes, pos, r_cut, prev, nxt = _pron_nodes(heard), 0, 0, None, None
    for i, n in enumerate(nodes):
        end = pos + len(n["string"])
        if cut < end:                          # lặng giữa một từ: chỉ hợp lệ nếu từ có ッ (一回 イッ|カイ)
            return "ッ" in n["r"]
        r_cut += len(n["r"])
        if end == cut:
            prev, nxt = n, (nodes[i + 1] if i + 1 < len(nodes) else None)
            break
        pos = end
    if prev is None or nxt is None:
        return True
    if prev["r"].endswith("ッ") or nxt["r"].startswith("ッ"):
        return True
    # đổi vị trí theo cách đọc của chữ ASR sang chữ gốc, xem có dấu câu ở đó không
    heard_r = "".join(n["r"] for n in nodes)
    orig_r, puncts = "", set()
    for n in _pron_nodes(re.sub(r"\{([^{}|]*)\|[^{}|]*\}", r"\1", text)):
        if re.fullmatch(r"[、，,。！？!?…「」『』（）()\s]+", n["string"]):
            puncts.add(len(orig_r))
        else:
            orig_r += n["r"]
    sm = difflib.SequenceMatcher(None, heard_r, orig_r, autojunk=False)
    for _, i1, i2, j1, j2 in sm.get_opcodes():
        if i1 <= r_cut <= i2:
            if any(abs(p - (j1 + min(r_cut - i1, j2 - j1))) <= 1 for p in puncts):
                return True
            break
    return not _tight_boundary(prev, nxt)


def find_unexpected_pauses(wav, sr: int, text: str, heard=None, min_pause: float = 0.15,
                           rel_threshold: float = 0.1):
    """Các chỗ ngắt nhịp bất thường giữa câu: lặng >= min_pause giây ở nơi chữ không có
    dấu câu và không phải âm ッ. VoxCPM2 thỉnh thoảng ngắt giữa cụm (三匹の… 猫と, hoặc
    ね… こと khi đầu vào là hiragana), nghe như ngập ngừng.
    heard: kết quả make_sensevoice_transcriber (có mốc thời gian) để định vị từng chỗ lặng;
    không có thì so số chỗ lặng với số dấu câu + số ッ. Trả [(giây, "trước|sau")]."""
    sil = _silences(wav, sr, min_pause, rel_threshold)
    times = getattr(heard, "times", None)
    if not times:
        body = re.sub(r"[。！？!?…]+$", "", text.strip())
        allowed = len(re.findall(r"[、，,。！？!?…]+", body)) + _reading(body).count("ッ")
        return [(round(a, 2), "?") for a, _ in sil[allowed:]]
    toks = [(tok.replace("▁", ""), t) for tok, t in times]
    full = "".join(tok for tok, _ in toks)
    out = []
    for a, b in sil:
        mid = (a + b) / 2
        cut = len("".join(tok for tok, t in toks if t < mid))
        if not _pause_allowed(full, cut, text):
            out.append((round(a, 2), f"{full[max(0, cut - 3):cut]}|{full[cut:cut + 3]}"))
    return out


def unexpected_pauses(wav, sr: int, text: str, heard=None, **kwargs) -> int:
    """Số chỗ ngắt nhịp bất thường (xem find_unexpected_pauses)."""
    return len(find_unexpected_pauses(wav, sr, text, heard, **kwargs))


def reading_problems(wav, sr: int, text: str, heard, pre=None, min_match: float = 0.85, max_extra: int = 2):
    """Các lỗi nghe ra trong một bản đọc (rỗng = đạt): đọc thiếu/sai (khớp < min_match),
    đọc thừa đầu/cuối câu, sai cụm số/từ đếm, sai một từ có kanji, ngắt nhịp giữa cụm.
    Trả (danh sách lỗi, độ khớp, số mora thừa)."""
    processed = (pre or JapaneseTextPreprocessor()).process(text)
    # so với cả câu gốc (ASR ghi 二十歳 cho ハタチ) lẫn câu đã áp từ điển (VoxCPM -> ボックスシーピーエム)
    match, extra = max(check_reading(text, heard), check_reading(processed, heard), key=lambda r: (r[0], -r[1]))
    problems = []
    if match < min_match:
        problems.append(f"khớp {match:.2f}")
    if extra > max_extra:
        problems.append(f"thừa {extra} mora")
    problems += [f"sai số {w}" for w in missing_counters(processed, heard)]
    problems += [f"sai từ {w}" for w in missing_words(processed, heard)]
    problems += [f"ngắt {where}" for _, where in find_unexpected_pauses(wav, sr, processed, heard)]
    return problems, match, extra


def synthesize_checked(model, text: str, ref_wav: str, ref_text: str, transcribe, pre=None, seed: int = 0,
                       max_tries: int = 3, min_match: float = 0.85, max_extra: int = 2, **kwargs):
    """Như synthesize(), nhưng nghe lại bằng ASR rồi sinh lại với seed khác khi
    reading_problems() thấy lỗi. Hết lượt thì trả bản ít lỗi nhất.
    Trả (wav, số lần thử, độ khớp)."""
    sr = model.tts_model.sample_rate
    best = None
    for attempt in range(max_tries):
        wav = synthesize(model, text, ref_wav, ref_text, pre=pre, seed=seed + attempt, **kwargs)
        heard = transcribe(wav, sr)
        problems, match, extra = reading_problems(wav, sr, text, heard, pre, min_match, max_extra)
        if not problems:
            return wav, attempt + 1, match
        score = (len(problems), -(match - 0.05 * extra))
        if best is None or score < best[0]:
            best = (score, wav, match)
    return best[1], max_tries, best[2]


if __name__ == "__main__":
    p = JapaneseTextPreprocessor()
    for s in ["私は学校へ行きます。本を読む。", "8月の誕生日に、三階で三匹の猫と一回だけ遊んだ。",
              "「本当ですか？」えっと……ｶﾞｯｺｳは１０時から！", "人気のない場所で待っていた。"]:
        print(s, "\n  ->", p.process(s), "\n  khó:", p.hard_reasons(s))
