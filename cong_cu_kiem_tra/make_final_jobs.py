"""Kiểm tra cuối: cấu hình tối ưu (ja_process_v2 mặc định) + cách đọc chỉ định, seed 2, toàn bộ 36 câu.

python make_final_jobs.py <ref_wav> <ref_transcript> <tên_biến_thể> [id ...]
"""
import json, sys
sys.path.insert(0, ".")
from ja_corpus import CORPUS
from ja_process_v2 import JapaneseTextPreprocessor
from make_fix_jobs_r2 import ANNOTATED

ANNOTATED = dict(ANNOTATED, d1="彼は「{明日|あした}また来ます」と言って、帰っていった。", h4="この{生物|せいぶつ}は水の中で生きています。")
ref, ref_text, variant = sys.argv[1], sys.argv[2], sys.argv[3]
only = set(sys.argv[4:])
pre = JapaneseTextPreprocessor()
jobs = [dict(name=f"{c[0]}_{variant}", id=c[0], sentence=c[2], text=pre.process(ANNOTATED.get(c[0], c[2])), seed=2,
             ref=ref, prompt_wav=ref, prompt_text=pre.process(ref_text))
        for c in CORPUS if not only or c[0] in only]
json.dump(jobs, open(f"jobs_{variant}.json", "w"), ensure_ascii=False, indent=1)
print(len(jobs), "job")
