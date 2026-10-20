"""Vòng 2: chạy bằng Ultimate Cloning (prompt) cho toàn bộ bộ câu + các phép so sánh.

python make_fix_jobs_r2.py [ref_wav]  ->  fix_jobs_r2.json
"""
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "user_pipeline")
from ja_corpus import CORPUS  # noqa: E402
from ja_particle_fix import fix_japanese_particles as fx  # noqa: E402

REF = "ref_ja_female.wav"
REF_TEXT = "こんにちわ。今日わ、私たちの町にできた新しい図書館について、少しお話ししたいと思います。"
S = {c[0]: c[2] for c in CORPUS}
ANNOTATED = {  # cách đọc chỉ định cho câu bộ phát hiện báo khó (giả lập bước LLM)
    "n2": "庭には二羽{鶏|ニワトリ}がいる。",
    "h1": "{人気|ひとけ}のない場所で、彼は一人で待っていた。",
    "l2": "{VoxCPM|ボックスシーピーエム}は日本語の音声合成ができます。",
    "h3": "{十分|じゅうぶん}に注意して、{上手|じょうず}に話してください。",
    "h2": "明日は雨が{降り|ふり}そうなので、傘を持って行きます。",
}


def src(sid):
    return ANNOTATED.get(sid, S[sid])


def job(sid, variant, text, seed=0, ref=REF):
    return dict(name=f"{sid}_{variant}" + (f"_s{seed}" if seed else ""), id=sid, sentence=S[sid], text=text,
                seed=seed, ref=ref, prompt_wav=ref, prompt_text=REF_TEXT)


def build(ref):
    from ab_test import user_pipeline

    jobs = [job(sid, "prompt", fx(src(sid)), ref=ref) for sid in S]                  # 1. toàn bộ câu, bản A
    jobs += [job(sid, "Yprompt", fx(src(sid), fix_particles=False), ref=ref)         # 2. giữ nguyên trợ từ
             for sid in ["p1", "p2", "p3", "p4", "q2", "k1", "k2", "s1"]]
    jobs += [job(sid, "Rprompt", S[sid], ref=ref) for sid in ["n1", "n3", "t1", "u2"]]  # 3. câu gốc
    jobs += [job(sid, "Uprompt", user_pipeline(S[sid]), ref=ref) for sid in ["n1", "p2", "n2"]]  # 4. pipeline cũ
    jobs += [job(sid, "Wprompt", fx(src(sid), convert_wo=False), ref=ref)            # 5. giữ を
             for sid in ["p2", "n4", "k2", "l1"]]
    return jobs


if __name__ == "__main__":
    jobs = build(sys.argv[1] if len(sys.argv) > 1 else REF)
    json.dump(jobs, open("fix_jobs_r2.json", "w"), ensure_ascii=False, indent=1)
    print(len(jobs), "job")
