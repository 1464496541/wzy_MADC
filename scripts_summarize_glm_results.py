"""Summarize glm-4-flashx math_500_id: checkpoint, cache key counts, majority vs GT."""
from __future__ import annotations

import json
from pathlib import Path

from wzy_multi_agent_debate_expand import get_majority_answer_from_expand, is_correct_answer
from wzy_multi_agent_debate_clustering import get_majority_answer_from_latest

ROOT = Path(__file__).resolve().parent / "glm-4-flashx"
DATA = ROOT / "data" / "math_500_id.json"
RES = ROOT / "results" / "debate_zy" / "math_500_id"
CK = ROOT / "checkpoint.json"
IS_MATH = True

FILES = {
    "expand": RES / "debate_zy_glm-4-flashx_10_1_expand_agent_com0_False.json",
    "exchange1": RES / "debate_zy_glm-4-flashx_10_1_exchange1_agent_com0_False.json",
    "exchange2": RES / "debate_zy_glm-4-flashx_10_1_exchange2_agent_com0_False.json",
    "bidirectional_1": RES
    / "debate_zy_glm-4-flashx_10_1_exchange_bidirectional_1_agent_com0_False.json",
    "bidirectional_2": RES
    / "debate_zy_glm-4-flashx_10_1_exchange_bidirectional_2_agent_com0_False.json",
}


def stage_acc(path: Path, use_latest: bool) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ok = tot = 0
    for v in data.values():
        if not isinstance(v, list) or len(v) < 3:
            continue
        ctxs, gt, _ = v[0], v[1], v[2]
        if use_latest:
            maj = get_majority_answer_from_latest(ctxs, is_math=IS_MATH)
        else:
            maj = get_majority_answer_from_expand(ctxs, is_math=IS_MATH)
        tot += 1
        if is_correct_answer(maj, gt, is_math=IS_MATH):
            ok += 1
    return ok, tot


def main() -> None:
    examples = json.loads(DATA.read_text(encoding="utf-8"))["examples"]
    all_ids = sorted(int(str(x["question_id"])) for x in examples)
    n_data = len(all_ids)

    ck = json.loads(CK.read_text(encoding="utf-8"))
    ck_done = sum(1 for v in ck.values() if v)
    ck_ids = sorted(int(k) for k in ck if str(k).isdigit())
    missing_ck = sorted(set(all_ids) - set(ck_ids))

    print("=" * 64)
    print("glm-4-flashx / math_500_id / FINAL SUMMARY")
    print("=" * 64)
    print(f"Dataset examples: {n_data} (question_id {all_ids[0]}..{all_ids[-1]})")
    print(f"checkpoint completed: {ck_done} / {n_data}")
    if ck_ids:
        print(f"checkpoint id span: {min(ck_ids)} .. {max(ck_ids)}")
    if missing_ck:
        head = missing_ck[:20]
        more = " ..." if len(missing_ck) > 20 else ""
        print(f"checkpoint missing ids ({len(missing_ck)}): {head}{more}")

    print()
    print("--- Cache files (top-level dict key count, file size) ---")
    for name, p in FILES.items():
        if not p.exists():
            print(f"{name}: MISSING {p.name}")
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        n = len(blob) if isinstance(blob, dict) else -1
        mb = p.stat().st_size / (1024 * 1024)
        print(f"{name:18s} keys={n:4d}  ~{mb:.1f} MB")

    print()
    print("--- Majority vote vs GT (math strip_string) ---")
    for name, p in FILES.items():
        if not p.exists():
            continue
        use_latest = name != "expand"
        ok, tot = stage_acc(p, use_latest)
        pct = 100.0 * ok / tot if tot else 0.0
        print(f"{name:18s} {ok}/{tot} = {pct:.2f}%")

    exp_path = FILES["expand"]
    if exp_path.exists():
        exp = json.loads(exp_path.read_text(encoding="utf-8"))
        ex_ids = {
            int(str(v[2]))
            for v in exp.values()
            if isinstance(v, list) and len(v) >= 3
        }
        miss_ex = sorted(set(all_ids) - ex_ids)
        print()
        print(f"expand distinct question_id in cache: {len(ex_ids)} / {n_data}")
        if miss_ex:
            head = miss_ex[:25]
            more = " ..." if len(miss_ex) > 25 else ""
            print(f"  missing ids ({len(miss_ex)}): {head}{more}")
        else:
            print("  all 500 dataset ids present in expand cache")

    print("=" * 64)


if __name__ == "__main__":
    main()
