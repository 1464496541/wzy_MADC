import os
os.environ["OMP_NUM_THREADS"] = "1"

"""
math_500 单题测试脚本

功能：
1. 从 math_500_id.json 中取出指定 question_id 的题目（默认第 1 题）
2. 对该题完整执行：expand -> Round 1 exchange -> Round 2 exchange
3. 在每个阶段打印 majority_answer 并与 ground_truth 对比
4. 所有子函数的 print 直接输出，便于人工核查中间结果
"""

import sys
import asyncio
import json
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ── 核心模块导入 ──
from wzy_multi_agent_debate_exchange import run_exchange_after_clustering
from wzy_multi_agent_debate_expand import MODEL_NAME, client, MODEL_TAG


# ============================================================
# 可配置项
# ============================================================

# 聚类方法："kmeans" 或 "dbscan"
USE_METHOD = "kmeans"

# exchange 轮数
NUM_ROUNDS = 2

# 指定测试的 question_id（整数或字符串均可）
# 设为 None 时默认取第 1 题
FIXED_QUESTION_ID = 80


# ============================================================
# 工具函数
# ============================================================

def _mark_str(correct: bool) -> str:
    return "correct" if correct else "wrong"


def _print_result(record: dict, num_rounds: int):
    """打印单题测试结果。"""
    q_id = record.get("question_id", "?")
    gt = record.get("ground_truth", "?")

    print("\n" + "=" * 60)
    print(f"  Q#{q_id}  标准答案: {gt}")
    print("=" * 60)

    if not record.get("success"):
        print(f"  [FAILED] {record.get('error', '未知错误')}")
        print("=" * 60)
        return

    expand_maj = record.get("expand_majority", "N/A")
    expand_ok = record.get("expand_correct", False)
    print(f"  Expand    阶段 : majority={expand_maj}  -> {_mark_str(expand_ok)}")

    for i, (maj, ok) in enumerate(
        zip(
            record.get("round_majorities", []),
            record.get("round_corrects", []),
        )
    ):
        print(f"  Exchange{i+1} 阶段 : majority={maj}  -> {_mark_str(ok)}")

    print("=" * 60)


# ============================================================
# 单题测试主入口
# ============================================================

async def run_single_test(
    use_method: str = USE_METHOD,
    num_rounds: int = NUM_ROUNDS,
    fixed_question_id=FIXED_QUESTION_ID,
):
    print("\n" + "=" * 60)
    print("# math_500 单题测试")
    print(f"#   聚类方法: {use_method.upper()}  |  Exchange 轮数: {num_rounds}")
    print("=" * 60)

    # ── 加载题目 ──
    model_name = MODEL_NAME
    task_file = f"{model_name}/data/math_500_id.json"
    print(f"\n[初始化] 模型: {model_name}  |  题目文件: {task_file}")
    with open(task_file, "r", encoding="utf-8") as f:
        all_items = json.load(f)["examples"]

    # ── 选取指定题目 ──
    if fixed_question_id is not None:
        target_id = str(fixed_question_id)
        matched = [it for it in all_items if str(it.get("question_id", "")) == target_id]
        if not matched:
            print(f"[错误] 未找到 question_id={target_id} 的题目，退出")
            return
        item = matched[0]
    else:
        item = all_items[0]

    q_id = str(item.get("question_id", "?"))
    print(f"[测试题目] question_id={q_id}")
    print(f"[题目内容] {item['input'][:200]}...")
    print(f"[标准答案] {item['target']}")

    # ── 执行完整流程 ──
    print(f"\n{'─'*60}")
    print("开始执行: expand -> exchange...")
    print(f"{'─'*60}\n")

    t0 = time.time()
    result = await run_exchange_after_clustering(
        use_method=use_method,
        num_rounds=num_rounds,
        question_item=item,
    )
    elapsed = time.time() - t0

    # ── 打印结果 ──
    _print_result(result, num_rounds)
    print(f"\n[耗时] {elapsed:.1f}s")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_single_test(
        use_method=USE_METHOD,
        num_rounds=NUM_ROUNDS,
        fixed_question_id=FIXED_QUESTION_ID,
    ))
