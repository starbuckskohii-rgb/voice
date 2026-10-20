"""Phát hiện câu tiếng Nhật "khó" (máy dễ đọc sai) để chỉ những câu đó mới nhờ LLM viết lại.

Ba tín hiệu, câu dính bất kỳ tín hiệu nào thì coi là khó:

1. DISAGREE  - UniDic và pyopenjtalk (hai từ điển độc lập) cho ra cách đọc khác
               nhau. Khi hai bên không thống nhất thì ít nhất một bên đang đoán:
               十分 (じゅっぷん/じゅうぶん), 明日 (あす/あした), ははは, わたしはははが...
2. HIDDEN_HA - Một chữ は/へ nằm trong cụm hiragana dài và bị gộp vào giữa một từ
               (đọc ha/he chứ không phải trợ từ), kiểu に|はにわ. Hai từ điển cùng sai
               giống nhau nên tín hiệu 1 không bắt được.
3. LONG_KANA - Cụm hiragana rất dài có chứa は/へ (ここではきものをぬいでください):
               có nhiều cách tách từ, máy chỉ chọn được một.
4. HETERONYM - Có từ kanji nằm trong danh sách từ nhiều cách đọc mà cả hai từ điển
               hay cùng đoán sai (人気: にんき/ひとけ, 一日: いちにち/ついたち...).
5. UNKNOWN   - Có từ không có trong từ điển (tên riêng, từ lóng viết hiragana...).

Cài đặt:  pip install fugashi unidic pyopenjtalk && python -m unidic download
"""

import re
import unicodedata

import pyopenjtalk

from ja_particle_fix import _has_digit, _is_hiragana, _kanji_to_kana_openjtalk, _tagger

_VOWEL_ROWS = {
    "ア": "アカガサザタダナハバパマヤラワァャヮ",
    "イ": "イキギシジチヂニヒビピミリィ",
    "ウ": "ウクグスズツヅヌフブプムユルゥュヴ",
    "エ": "エケゲセゼテデネヘベペメレェ",
    "オ": "オコゴソゾトドノホボポモヨロヲォョ",
}
_VOWEL_OF = {k: v for v, row in _VOWEL_ROWS.items() for k in row}

# Khác biệt đã biết là vô hại giữa hai từ điển (cùng một cách phát âm).
_EQUIV = [("トユー", "トイウ"), ("ワタクシ", "ワタシ"), ("ホントー", "ホント"), ("ホントウ", "ホント"), ("タンジョーヒ", "タンジョービ"), ("ヲ", "オ"), ("ニッポン", "ニホン")]

# Cụm hiragana dài tối thiểu bao nhiêu chữ thì mới xét tín hiệu 2.
MIN_HIRAGANA_RUN = 5
# Cụm hiragana có は/へ dài từ bao nhiêu chữ thì coi là khó (tín hiệu 3).
LONG_KANA_RUN = 10

# Từ có nhiều cách đọc phổ biến, nghĩa phụ thuộc ngữ cảnh. Bổ sung theo nội dung của bạn.
HETERONYMS = {
    "人気", "一日", "十分", "鶏", "上手", "下手", "生物", "市場",
    "大家", "風車", "色紙", "一行", "寒気", "最中", "何人", "何分", "目下", "上下",
    "分別", "心中", "大事", "工夫", "仮名", "末期", "見物", "金星", "変化", "利益",
}


def _normalize_pron(p: str) -> str:
    """Bỏ dấu câu và khác biệt trường âm (ー/イ/ウ) để chỉ so phần phát âm thật."""
    p = unicodedata.normalize("NFKC", p)
    for a, b in _EQUIV:
        p = p.replace(a, b)
    out = []
    for c in p:
        if not ("ァ" <= c <= "ー"):
            continue  # dấu câu, ký tự Latin...
        prev_vowel = _VOWEL_OF.get(out[-1]) if out else None
        if c == "ー" and prev_vowel:
            c = prev_vowel
        elif (c == "イ" and prev_vowel == "エ") or (c == "ウ" and prev_vowel == "オ"):
            c = prev_vowel  # センセイ == センセエ, トウキョウ == トオキョオ
        out.append(c)
    return "".join(out)


def _pron_unidic(text: str) -> str:
    return "".join(w.feature.pron if w.feature.pron and w.feature.pron != "*" else w.surface for w in _tagger(text))


def _pron_openjtalk(text: str) -> str:
    return "".join(n["pron"].replace("’", "") for n in pyopenjtalk.run_frontend(text))


def _diff_spans(a: str, b: str):
    import difflib

    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return [(a[i1:i2], b[j1:j2]) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]


def find_hard_reasons(text: str) -> list:
    """Trả về danh sách lý do câu khó; rỗng nghĩa là câu dễ, xử lý bằng script là đủ."""
    reasons = []

    # Số đã được ja_particle_fix đọc bằng luật của pyopenjtalk; đổi trước sang katakana
    # để UniDic (không đọc được chữ số) không bị tính là "bất đồng".
    cmp_text = _kanji_to_kana_openjtalk(text, numbers_only=True, number_script="katakana")
    a = _normalize_pron(_pron_unidic(cmp_text))
    b = _normalize_pron(_pron_openjtalk(cmp_text))
    if a != b:
        reasons.append(("DISAGREE", _diff_spans(a, b)))
    for w in _tagger(cmp_text):
        # katakana luôn đọc đúng theo mặt chữ nên từ lạ viết katakana không phải rủi ro
        if w.is_unk and not _has_digit(w.surface) and not all("ァ" <= c <= "ー" for c in w.surface):
            reasons.append(("UNKNOWN", w.surface))

    hiragana_runs = [(m.start(), m.end()) for m in re.finditer(r"[ぁ-ゟ]+", text)]
    for rs, re_ in hiragana_runs:
        run = text[rs:re_]
        if re_ - rs >= LONG_KANA_RUN and ("は" in run or "へ" in run):
            reasons.append(("LONG_KANA", run))

    for word in HETERONYMS:
        if word in text:
            reasons.append(("HETERONYM", word))

    pos = 0
    for w in _tagger(text):
        pos += len(w.white_space)
        start, end = pos, pos + len(w.surface)
        pos = end
        s = w.surface
        pron = w.feature.pron or ""
        if len(s) < 2 or not _is_hiragana(s) or len(s) != len(pron):
            continue
        hidden = [c for c, p in zip(s, pron) if (c, p) in (("は", "ハ"), ("へ", "ヘ"))]
        in_long_run = any(rs <= start and end <= re_ and re_ - rs >= MIN_HIRAGANA_RUN for rs, re_ in hiragana_runs)
        if hidden and in_long_run:
            reasons.append(("HIDDEN_HA", s))

    return reasons


def is_hard(text: str) -> bool:
    return bool(find_hard_reasons(text))
