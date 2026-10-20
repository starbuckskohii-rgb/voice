"""Tiền xử lý tiếng Nhật trước khi đưa vào VoxCPM.

VoxCPM không có bước G2P (chữ -> âm) cho tiếng Nhật: model tự đoán cách đọc từ
chữ viết. Script sửa 3 nhóm lỗi:

1. Trợ từ: は/へ/を viết một kiểu, đọc wa/e/o. Dùng MeCab (fugashi + UniDic), so
   từng chữ với cách PHÁT ÂM (pron): chữ viết は mà phát âm ワ thì đổi thành わ
   (tương tự へ->え, を->お). Bắt được cả こんにちは, あるいは, もしくは; không
   đụng vào はい, ははは, へや...
2. Số + từ đếm + ngày tháng: dùng pyopenjtalk (có luật biến âm: 三階 さんがい,
   三匹 さんびき, 一回 いっかい, 4月1日 しがつついたち, 60% ろくじゅっパーセント).
3. Cách đọc chỉ định tay: READING_FIX và cú pháp {人気|ひとけ} (từ LLM hoặc bạn).

Phần còn lại GIỮ NGUYÊN kanji: đổi hết sang kana làm VoxCPM mất ranh giới từ
(さんびきのねこと bị đọc "none koto").

Cài đặt:  pip install fugashi unidic pyopenjtalk && python -m unidic download
"""

import re

import fugashi

try:  # pyopenjtalk đọc số đếm (三本, 一本, 二羽, 2026年...) đúng hơn UniDic nhiều
    import pyopenjtalk
except ImportError:
    pyopenjtalk = None

try:  # ưu tiên UniDic bản đầy đủ nếu đã cài
    import unidic

    _tagger = fugashi.Tagger("-d " + unidic.DICDIR)
except Exception:
    _tagger = fugashi.Tagger()

# chữ viết -> (âm thực tế trong pron, chữ thay thế)
_TARGETS = {"は": ("ワ", "わ"), "へ": ("エ", "え"), "を": ("オ", "お")}

# Thay cả cụm TRƯỚC khi phân tích: dùng cho câu toàn hiragana mà máy tách từ sai.
PHRASE_FIX = {
    "にわにはにわにわとり": "庭には二羽鶏",
}

# Cách đọc chỉ định tay (từ dài được ưu tiên trước). Thêm tuỳ ý theo nội dung.
READING_FIX = {
    "二十歳": "はたち",
    "20歳": "はたち",
    "鶏": "ニワトリ",  # 二羽鶏: model bỏ mất một "にわ" khi để kanji; katakana thì đọc đủ
    # "鶏": "にわとり",
    # "人気のない": "ひとけのない",
}

_KANJI_NUMERALS = set("〇一二三四五六七八九十百千万億兆")
# pyopenjtalk đổi chữ Latin/số ASCII sang full-width (ＶｏｘＣＰＭ); đổi ngược lại,
# nhưng giữ nguyên dấu câu tiếng Nhật (？！……「」).
_FULLWIDTH_ALNUM = {chr(c): chr(c - 0xFEE0) for r in ((0xFF10, 0xFF1A), (0xFF21, 0xFF3B), (0xFF41, 0xFF5B)) for c in range(*r)}
_FULLWIDTH_ALNUM["　"] = " "
_PLACEHOLDER_BASE = 0xE000  # ký tự Private Use để giữ chỗ cho cách đọc chỉ định tay


def _is_hiragana(s: str) -> bool:
    return all("ぁ" <= c <= "ゟ" for c in s)


def _has_kanji(s: str) -> bool:
    return any("一" <= c <= "鿿" or c == "々" for c in s)


def _has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


def _kata_to_hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def _hira_to_kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def _safe_kana(reading: str, force_katakana: bool = False) -> str:
    """Kana chèn vào câu: hiragana, trừ khi có âm ハ/ヘ thì dùng katakana để VoxCPM
    không tưởng chữ は/へ đó là trợ từ (履物 -> ハキモノ, 八本 -> ハッポン)."""
    kata = _hira_to_kata(reading)
    if force_katakana or "ハ" in kata or "ヘ" in kata:
        return kata
    return _kata_to_hira(kata)


def _fix_token(surface: str, pron, convert_wo: bool) -> str:
    if not pron or not _is_hiragana(surface) or len(surface) != len(pron):
        return surface
    chars = list(surface)
    for i, c in enumerate(chars):
        if c in _TARGETS and (c != "を" or convert_wo):
            spoken, replacement = _TARGETS[c]
            if pron[i] == spoken:
                chars[i] = replacement
    return "".join(chars)


def _node_reading(node) -> str:
    """Lấy cách đọc từ pron (có biến âm: 三本->サンボン, 一本->イッポン) nhưng giữ
    nguyên âm dài theo read (トウキョウ thay vì トーキョー)."""
    read = node["read"]
    pron = node["pron"].replace("’", "")
    if len(read) != len(pron):
        return pron
    return "".join(r if p == "ー" else p for r, p in zip(read, pron))


def _is_number_node(node) -> bool:
    """Số (三, 千, 8), từ đếm (匹, 階, 回, %, km) và từ chỉ ngày/người có chữ số
    (一日 ついたち, 二日 ふつか, 一人 ひとり, 4月 しがつ)."""
    s = node["string"]
    return (
        node["pos_group1"] == "数"
        or node["pos_group2"] == "助数詞"
        or _has_digit(s)
        or (node["pos_group1"] == "副詞可能" and any(c in _KANJI_NUMERALS for c in s))
    )


# Ký hiệu đơn vị sau số mà VoxCPM2 đọc hỏng (500km -> "500件の犬"). % thì đọc đúng nên giữ.
UNIT_READINGS = {
    "km": "キロメートル", "kg": "キログラム", "cm": "センチメートル", "mm": "ミリメートル",
    "mg": "ミリグラム", "ml": "ミリリットル", "mL": "ミリリットル", "m": "メートル", "g": "グラム",
    "L": "リットル", "℃": "度", "°C": "度",
}
_UNIT_RE = re.compile(r"(?<=[0-9０-９])(\s?)(" + "|".join(sorted(map(re.escape, UNIT_READINGS), key=len, reverse=True))
                      + r")(?![A-Za-z])")


def _replace_units(text: str) -> str:
    """Chỉ thay ký hiệu đơn vị đứng ngay sau số; giữ nguyên mọi ký tự khác."""
    return _UNIT_RE.sub(lambda m: UNIT_READINGS[m.group(2)], text)


def _kanji_to_kana_openjtalk(text: str, numbers_only: bool, number_script: str) -> str:
    """Đổi các cụm số (hoặc mọi kanji nếu numbers_only=False) sang kana; phần còn lại
    giữ nguyên chữ gốc (để pyopenjtalk không đụng lại vào は/へ đã xử lý)."""
    out, group = [], []

    def flush():
        if not group:
            return
        reading = "".join(_node_reading(n) for n in group)
        # đơn vị viết bằng ký hiệu (%, km) là từ ngoại lai -> katakana
        has_symbol_unit = any(not _has_kanji(n["string"]) and not _has_digit(n["string"]) for n in group)
        out.append(_safe_kana(reading, force_katakana=has_symbol_unit or number_script == "katakana"))
        group.clear()

    for node in pyopenjtalk.run_frontend(text):
        s = node["string"]
        if _is_number_node(node) if numbers_only else (_has_kanji(s) or _has_digit(s)):
            group.append(node)
            continue
        flush()
        out.append("".join(_FULLWIDTH_ALNUM.get(c, c) for c in s))
    flush()
    return "".join(out)


def _is_punct(s: str) -> bool:
    import unicodedata

    return all(unicodedata.category(c).startswith(("P", "Z")) for c in s)


def _split_bunsetsu(text: str) -> list:
    """Chia câu GỐC thành các cụm ngữ điệu (bunsetsu) theo pyopenjtalk: 三匹の|猫と|...
    Phải chia trước khi đổi trợ từ, vì 本お読む sẽ bị hiểu お là tiền tố kính ngữ."""
    chunks = []
    for node in pyopenjtalk.run_frontend(text):
        s = "".join(_FULLWIDTH_ALNUM.get(c, c) for c in node["string"])
        if chunks and (node["chain_flag"] != 0 or _is_punct(s)):
            chunks[-1] += s
        else:
            chunks.append(s)
    return chunks


def fix_japanese_particles(
    text: str,
    convert_wo: bool = True,
    kana_mode: str = "numbers",
    bunsetsu_spaces: bool = False,
    fix_particles: bool = True,
    number_script: str = "hiragana",
) -> str:
    """kana_mode:
    - "units": chỉ đổi ký hiệu đơn vị (km -> キロメートル, kg, cm). Khuyên dùng khi gọi
      VoxCPM2 bằng Ultimate Cloning (prompt tiếng Nhật): model tự đọc đúng số đếm/ngày.
    - "numbers": đổi số + từ đếm + ngày sang kana, giữ kanji cho từ khác. Dùng khi
      KHÔNG có prompt tiếng Nhật. Cần pyopenjtalk.
    - "all": đổi mọi kanji sang kana. Đọc đúng âm hơn nhưng dễ chia cụm sai.
    - "none": giữ nguyên kanji, chỉ sửa trợ từ.
    number_script: "hiragana" (さんびき) hoặc "katakana" (サンビキ) cho cụm số.
    fix_particles=False: giữ nguyên は/へ/を (để A/B test).
    bunsetsu_spaces=True: chèn khoảng trắng giữa các cụm ngữ điệu (thử nghiệm).
    """
    for phrase, replacement in PHRASE_FIX.items():
        text = text.replace(phrase, replacement)

    if bunsetsu_spaces and pyopenjtalk is not None:
        out = ""
        for chunk in _split_bunsetsu(text):
            fixed = fix_japanese_particles(chunk, convert_wo, kana_mode, False, fix_particles, number_script)
            out += fixed if not out or _is_punct(out[-1]) else " " + fixed
        return out

    # Cách đọc chỉ định tay được giữ chỗ bằng ký tự riêng, điền vào SAU bước sửa trợ
    # từ, để はたち không bị bước đó hiểu nhầm thành trợ từ.
    readings = []

    def hold(reading):
        # katakana chỉ định tay (ニワトリ) giữ nguyên; hiragana thì qua _safe_kana
        is_kata = all("ァ" <= c <= "ー" for c in reading)
        readings.append(reading if is_kata else _safe_kana(reading))
        return chr(_PLACEHOLDER_BASE + len(readings) - 1)

    text = re.sub(r"\{[^{}|]*\|([^{}|]*)\}", lambda m: hold(m.group(1)), text)
    for word in sorted(READING_FIX, key=len, reverse=True):
        if word not in text:
            continue
        # chỉ thay khi khớp đúng ranh giới từ (鶏 trong 二羽鶏, không phải 鶏肉/親鶏/若鶏)
        # và không đứng sau chữ số (120歳, 百二十歳 vẫn đọc theo số)
        bounds, pos = {0, len(text)}, 0
        for w in _tagger(text):
            pos = text.find(w.surface, pos)
            bounds.update((pos, pos + len(w.surface)))
            pos += len(w.surface)
        parts, last = [], 0
        for m in re.finditer(r"(?<![0-9０-９〇一二三四五六七八九十百千万])" + re.escape(word), text):
            if m.start() in bounds and m.end() in bounds:
                parts += [text[last:m.start()], hold(READING_FIX[word])]
                last = m.end()
        text = "".join(parts) + text[last:]

    if kana_mode == "units":
        text = _replace_units(text)
    use_openjtalk = kana_mode in ("numbers", "all") and pyopenjtalk is not None
    out = []
    for word in _tagger(text):
        surface = word.surface
        kana = word.feature.kana
        if kana_mode == "all" and not use_openjtalk and _has_kanji(surface) and kana and kana != "*":
            surface = _kata_to_hira(kana)
        if fix_particles:
            surface = _fix_token(surface, word.feature.pron, convert_wo)
        out.append(word.white_space + surface)
    text = "".join(out)

    for i, reading in enumerate(readings):
        text = text.replace(chr(_PLACEHOLDER_BASE + i), reading)

    if use_openjtalk:
        text = _kanji_to_kana_openjtalk(text, numbers_only=(kana_mode == "numbers"), number_script=number_script)
    return text


if __name__ == "__main__":
    samples = [
        "私は学生です。",
        "明日、東京へ行きます。",
        "こんにちは、今日はいい天気ですね。",
        "8月の誕生日に、三階で三匹の猫と一回だけ遊んだ。",
        "会議は4月1日から一週間です。",
        "気温は25度で、湿度は60%です。",
        "一人で二十歳の誕生日を祝った。",
        "「本当ですか？」えっと……すごい！",
    ]
    for s in samples:
        print(f"{s}\n  -> {fix_japanese_particles(s)}\n")
