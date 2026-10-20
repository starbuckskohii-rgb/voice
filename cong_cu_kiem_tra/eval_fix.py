"""Chấm các job sửa lỗi trong fix_out và so với bản A không giọng mẫu (ab_out).

Chạy:  python eval_fix.py fix_out [ab_out]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_all import evaluate, score_keywords  # noqa: E402
from ja_corpus import CORPUS  # noqa: E402

EXPECTED = {c[0]: c[3] for c in CORPUS}


def load_eval(out_dir, manifest_rows):
    ev_path = os.path.join(out_dir, "eval.jsonl")
    done = {}
    if os.path.exists(ev_path):
        done = {json.loads(l)["file"]: json.loads(l) for l in open(ev_path)}
    res = []
    for row in manifest_rows:
        if row["file"] not in done:
            r = evaluate(row)
            with open(ev_path, "a") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done[row["file"]] = r
        res.append(score_keywords(done[row["file"]]))
    return res


def main():
    fix_dir = sys.argv[1]
    ab_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "ab_out")
    rows = []
    for l in open(os.path.join(fix_dir, "manifest.jsonl")):
        r = json.loads(l)
        if "id" not in r:
            continue
        r["variant"] = r["name"].split("_", 1)[1].replace("_s", "@s")
        r["expected"] = EXPECTED[r["id"]]
        rows.append(r)
    results = load_eval(fix_dir, rows)
    ab = {}
    if os.path.exists(os.path.join(ab_dir, "eval.jsonl")):
        for l in open(os.path.join(ab_dir, "eval.jsonl")):
            r = json.loads(l)
            if r["variant"] == "A" and r["seed"] == 0:
                ab[r["id"]] = score_keywords(r)

    def line(tag, r):
        bad = [p["particle"] for p in r["particles"] if p["misread"]]
        ok = "✅" if not r["kw_fail"] and not bad else "❌"
        return (f"  {ok} {tag:7} sai={r['kw_fail'] or '-'} trợ từ sai={bad or '-'} {r['mora_per_sec']} mora/s\n"
                f"           R: {r['asr']['Reazon']}\n           S: {r['asr']['SenseV']}")

    by_id = {}
    for r in results:
        by_id.setdefault(r["id"], []).append(r)
    score = {}
    for sid, rs in by_id.items():
        print(f"\n### {sid}: {rs[0]['sentence']}  (đúng: {rs[0]['expected']})")
        if sid in ab:
            print(line("noref", ab[sid]))
            score.setdefault("noref", []).append(not ab[sid]["kw_fail"])
        for r in rs:
            print(line(r["variant"], r))
            score.setdefault(r["variant"], []).append(not r["kw_fail"] and not any(p["misread"] for p in r["particles"]))
    print("\n===== Số câu đạt (đọc đúng mọi từ khoá, không sai trợ từ) =====")
    for k, v in score.items():
        print(f"  {k:7} {sum(v)}/{len(v)}")


if __name__ == "__main__":
    main()
