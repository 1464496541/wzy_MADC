import os
os.environ["OMP_NUM_THREADS"] = "1"

"""
math_500 全量批量测试脚本

功能：
1. 遍历 math_500_id.json 中的所有题目（默认 500 道）
2. 对每道题完整执行：expand -> Round 1 exchange -> Round 2 exchange
3. 在每个阶段提取 majority_answer 并与 ground_truth 对比
4. 将每道题的结果实时追加写入 JSONL 结果文件（支持断点续跑）
5. 所有题目完成后打印三个阶段的正确率汇总

输出模式（由 VERBOSE 控制）：
- VERBOSE = True  调试模式：子函数内所有 print 正常输出，便于人工核查中间结果
  建议搭配 FIXED_QUESTION_ID 先跑单题，确认流程和结果正确后再切换到批量模式
- VERBOSE = False 批量模式：使用 contextlib.redirect_stdout 将子函数内所有 print
  重定向到内存缓冲区，控制台只保留批量层的进度信息（题号、耗时、各阶段正确性）
  发生异常时捕获到的静默日志会输出到 stderr 供排查
"""

import sys
import asyncio
import contextlib
import io
import json
import time
import traceback
from datetime import datetime

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

# exchange 轮数（与 run_exchange_after_clustering 默认保持一致）
NUM_ROUNDS = 2

# 结果输出目录（自动创建）
RESULT_DIR = "batch_results"

# 指定单题测试的 question_id（整数或字符串均可）
# 设为 None 时为批量模式，跑全部 500 道题
# 设为整数如 355 时：只测试该 question_id 对应的题目
FIXED_QUESTION_ID = None

# 相邻两道题之间的间隔秒数（避免触发 API 速率限制）
INTER_QUESTION_SLEEP = 2

# 输出模式：
#   True  = 调试模式，子函数内所有 print 正常输出，便于人工核查中间结果
#           建议搭配 FIXED_QUESTION_ID 先单题验证，确认无误后再切批量模式
#   False = 批量模式，子函数输出全部静默，控制台只显示批量层进度信息
VERBOSE = False

# ============================================================
# 工具函数
# ============================================================

def _result_file_path(model_name: str) -> str:
    """生成断点续跑用的 JSONL 结果文件路径（存放于 RESULT_DIR）。"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    return os.path.join(RESULT_DIR, f"{safe_name}_math500_{USE_METHOD}.jsonl")


def _debate_zy_dir(model_name: str) -> str:
    """返回与 expand 模块一致的 debate_zy 结果目录，并自动创建。"""
    d = os.path.join(model_name, "results", "debate_zy", "math_500_id")
    os.makedirs(d, exist_ok=True)
    return d



def _stage_contexts_file_path(model_name: str, stage: str) -> str:
    """生成指定阶段的上下文缓存文件路径（存放于模型的 results/debate_zy 目录下）。

    Args:
        stage: 阶段名，如 "expand" / "round1" / "round2"
    """
    return os.path.join(
        _debate_zy_dir(model_name),
        f"batch_{USE_METHOD}_{stage}_contexts.json",
    )


def _save_stage_contexts(
    cache_file: str,
    question_id: str,
    agent_contexts: list,
    ground_truth: str,
    question_text: str = None,
):
    """将单道题指定阶段的 agent_contexts 增量保存到缓存文件。

    三个阶段（expand / round1 / round2）统一以题目文本（question_text）作为键，
    与 debate_bbh_qwen3b.py 的读取方式 results[question][0][agent_idx][2]["content"]
    对齐，便于两侧直接共用缓存文件：
    {
      "<question_text>": [agent_contexts, ground_truth, question_id]
    }

    question_text 为 None 时（兜底）退化为 question_id 字符串作为键。

    使用 indent=2 保证文件易读；每次调用先读取已有缓存再合并写回。
    """
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}

    key = question_text if question_text is not None else str(question_id)
    cache[key] = [agent_contexts, ground_truth, str(question_id)]

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def _load_done_ids(result_file: str) -> set:
    """
    从已有结果文件中读取已成功完成的 question_id 集合（支持断点续跑）。

    只将 success=True 的记录视为"已完成"并跳过；
    success=False 的失败记录不计入，下次运行时会自动重跑这些题目。
    """
    done = set()
    if not os.path.exists(result_file):
        return done
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                # 只有 success=True 的记录才算已完成，失败的题目下次会重跑
                if record.get("success") and record.get("question_id"):
                    done.add(str(record["question_id"]))
            except json.JSONDecodeError:
                pass
    return done


def _append_result(result_file: str, record: dict):
    """将单题结果追加写入 JSONL 文件（每行一个 JSON 对象）。"""
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _mark_str(correct: bool) -> str:
    return "correct" if correct else "wrong"


def _print_summary(records: list, num_rounds: int):
    """打印所有已完成题目的三阶段正确率汇总。"""
    total = len(records)
    if total == 0:
        print("[汇总] 无有效记录。")
        return


    # success 来自 run_exchange_after_clustering 的返回值，它只回答一个问题：
    # 整个流程（expand + 2轮exchange）有没有报错、有没有中途崩溃
    success_records = [r for r in records if r.get("success")]
    failed_records  = [r for r in records if not r.get("success")]

    expand_correct = sum(1 for r in success_records if r.get("expand_correct"))
    round_corrects = [0] * num_rounds
    round_totals   = [0] * num_rounds
    for r in success_records:
        for i, rc in enumerate(r.get("round_corrects", [])):
            if i < num_rounds:
                round_totals[i] += 1
                if rc:
                    round_corrects[i] += 1

    n_success = len(success_records)
    print("\n" + "=" * 70)
    print(f"[批量测试汇总]  共 {total} 道题  |  成功 {n_success}  |  失败 {len(failed_records)}")
    print("=" * 70)
    print(f"  Expand   阶段正确率 : {expand_correct}/{n_success}"
          f"  ({expand_correct/n_success*100:.1f}%)" if n_success else "  Expand 阶段: 无数据")
    for i in range(num_rounds):
        n = round_totals[i]
        c = round_corrects[i]
        label = f"  Round {i+1} 阶段正确率 :"
        print(f"{label} {c}/{n}  ({c/n*100:.1f}%)" if n else f"{label} 无数据")
    if failed_records:
        print(f"\n[失败题目明细]  共 {len(failed_records)} 道")
        print("-" * 70)
        for idx, r in enumerate(failed_records, start=1):
            q_id = r.get("question_id", "?")
            gt   = r.get("ground_truth", "?")
            err  = r.get("error", "未知错误")
            print(f"  {idx}. Q#{q_id}  (标准答案: {gt})")
            for line in err.strip().splitlines():
                print(f"     {line}")
            if idx < len(failed_records):
                print()
        print("-" * 70)
    print("=" * 70)


# ============================================================
# 单题处理（静默模式）
# ============================================================

async def _process_one_question(
    item: dict,
    use_method: str,
    num_rounds: int,
    verbose: bool = VERBOSE,
) -> dict:
    """
    对单道题目执行完整流程，返回结构化结果。

    Args:
        item: 题目字典，含 input / target / question_id
        use_method: 聚类方法，"kmeans" 或 "dbscan"
        num_rounds: exchange 轮数
        verbose: 输出模式
            True  = 调试模式，子函数内所有 print 直接输出到控制台，便于人工核查
            False = 批量模式，子函数输出重定向到内存缓冲区静默；
                    发生异常时将捕获的日志输出到 stderr 供排查
    """
    if verbose:
        # ── 调试模式：直接调用，不做任何重定向 ──
        try:
            result = await run_exchange_after_clustering(
                use_method=use_method,
                num_rounds=num_rounds,
                question_item=item,
            )
            return result
        except Exception as e:
            return {
                "question_id": str(item.get("question_id", "?")),
                "ground_truth": item.get("target"),
                "expand_majority": None,
                "expand_correct": False,
                "round_majorities": [],
                "round_corrects": [],
                "expand_agent_contexts": None,
                "expand_agent_replies": [],
                "round_agent_replies": [],
                "round_agent_contexts": [],
                "success": False,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            }
    else:
        # ── 批量模式：redirect_stdout 静默所有子函数 print ──
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                result = await run_exchange_after_clustering(
                    use_method=use_method,
                    num_rounds=num_rounds,
                    question_item=item,
                )
            return result
        except Exception as e:
            # 将静默期间捕获的日志输出到 stderr，便于排查
            captured = buf.getvalue()
            if captured:
                print(f"\n[静默日志 - Q#{item.get('question_id')}]\n{captured}", file=sys.stderr)
            return {
                "question_id": str(item.get("question_id", "?")),
                "ground_truth": item.get("target"),
                "expand_majority": None,
                "expand_correct": False,
                "round_majorities": [],
                "round_corrects": [],
                "expand_agent_contexts": None,
                "expand_agent_replies": [],
                "round_agent_replies": [],
                "round_agent_contexts": [],
                "success": False,
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            }


# ============================================================
# 批量主入口
# ============================================================

async def run_batch_test_math500(
    use_method: str = USE_METHOD,
    num_rounds: int = NUM_ROUNDS,
    fixed_question_id=FIXED_QUESTION_ID,
):
    """
    批量测试 math_500 全部题目。

    Args:
        use_method: 聚类方法，"kmeans" 或 "dbscan"
        num_rounds: 每道题 exchange 的轮数
        fixed_question_id: 指定单题测试的 question_id；不为 None 时为单题模式，否则跑全部 500 道
    """
    start_time = time.time()
    print("\n" + "=" * 70)
    print("# math_500 批量测试")
    print(f"#   聚类方法: {use_method.upper()}  |  Exchange 轮数: {num_rounds}")
    print(f"#   启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── API 验证（只做一次，直接使用 expand 模块的 client）──
    print("\n[初始化] 验证 API 可用性...")
    try:
        client.chat.completions.create(
            model=MODEL_TAG,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        print("[API 验证] 通过")
    except Exception as _api_err:
        print(f"[错误] API 验证失败: {_api_err}，退出")
        return

    # ── 加载题目数据（model_name 直接使用 expand 模块的 MODEL_NAME 常量）──
    model_name = MODEL_NAME
    task_file = f"{model_name}/data/math_500_id.json"
    print(f"[初始化] 模型: {model_name}  |  题目文件: {task_file}")
    with open(task_file, "r", encoding="utf-8") as f:
        all_items = json.load(f)["examples"]

    if fixed_question_id is not None:
        target_id = str(fixed_question_id)
        all_items = [it for it in all_items if str(it.get("question_id", "")) == target_id]
        if not all_items:
            print(f"[错误] 未找到 question_id={target_id} 的题目，退出")
            return
        print(f"[配置] 单题模式：question_id={target_id}")

    total = len(all_items)
    print(f"[初始化] 共 {total} 道题待测试")

    # ── 断点续跑：读取已完成的题目 ──
    result_file = _result_file_path(model_name)
    done_ids = _load_done_ids(result_file)
    if done_ids:
        print(f"[断点续跑] 检测到结果文件，已完成 {len(done_ids)} 道题，跳过这些题目")
    print(f"[断点续跑文件] {result_file}")
    print(f"[Expand 上下文] {_stage_contexts_file_path(model_name, 'expand')}")
    for _rnd in range(1, num_rounds + 1):
        print(f"[Round {_rnd} 上下文] {_stage_contexts_file_path(model_name, f'round{_rnd}')}")
    print()

    # ── 逐题处理 ──
    all_records = []

    # 程序重启时，从已有 JSONL 结果文件中读取历史记录加入汇总（用于最终统计）
    if done_ids and os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # 过滤掉已完成的题目
    pending_items = [it for it in all_items if str(it.get("question_id", "")) not in done_ids]
    print(f"[进度] 待处理: {len(pending_items)} 道题")

    # 大体积字段不写入 JSONL（已单独保存到上下文缓存文件）
    _large_fields = {
        "expand_agent_contexts",
        "expand_agent_replies",
        "round_agent_replies",
        "round_agent_contexts",
    }

    for seq, item in enumerate(pending_items, start=1):
        q_id = str(item.get("question_id", "?"))
        global_seq = len(done_ids) + seq  # 总进度（含已完成）

        t0 = time.time()
        print(
            f"\n[{global_seq:>3}/{total}] Q#{q_id:>4}  "
            f"({datetime.now().strftime('%H:%M:%S')})  处理中...",
            end="",
            flush=True,
        )

        record = await _process_one_question(item, use_method, num_rounds, verbose=VERBOSE)

        elapsed = time.time() - t0

        # 打印本题进度行
        if record.get("success"):
            e_mark  = _mark_str(record["expand_correct"])
            r_marks = "  ".join(
                f"r{i+1}={_mark_str(rc)}"
                for i, rc in enumerate(record.get("round_corrects", []))
            )
            print(
                f"\r[{global_seq:>3}/{total}] Q#{q_id:>4}  "
                f"expand={e_mark}  {r_marks}  [{elapsed:.1f}s]"
            )
        else:
            err_short = str(record.get("error", ""))[:80].replace("\n", " ")
            print(
                f"\r[{global_seq:>3}/{total}] Q#{q_id:>4}  "
                f"[FAILED] {err_short}  [{elapsed:.1f}s]"
            )

        # 分阶段保存完整 agent 上下文（三阶段统一以题目文本为键）
        if record.get("success"):
            gt = record.get("ground_truth", "")
            question_text = item["input"]
            if record.get("expand_agent_contexts"):
                _save_stage_contexts(
                    _stage_contexts_file_path(model_name, "expand"),
                    q_id, record["expand_agent_contexts"], gt,
                    question_text=question_text,
                )
            for rnd_idx, rnd_ctx in enumerate(record.get("round_agent_contexts", [])):
                _save_stage_contexts(
                    _stage_contexts_file_path(model_name, f"round{rnd_idx + 1}"),
                    q_id, rnd_ctx, gt,
                    question_text=question_text,
                )

        # 写入断点续跑 JSONL 时剥离大体积字段（已单独保存到上下文缓存文件）
        record_for_jsonl = {k: v for k, v in record.items() if k not in _large_fields}
        _append_result(result_file, record_for_jsonl)
        all_records.append(record_for_jsonl)

        # 题目间等待，缓解 API 速率压力
        if seq < len(pending_items):
            await asyncio.sleep(INTER_QUESTION_SLEEP)

    # ── 最终汇总 ──
    total_elapsed = time.time() - start_time
    print(f"\n[完成] 全部题目处理完毕，总耗时 {total_elapsed/60:.1f} 分钟")
    _print_summary(all_records, num_rounds)
    print(f"[结果文件] {result_file}")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    asyncio.run(run_batch_test_math500(
        use_method=USE_METHOD,
        num_rounds=NUM_ROUNDS,
        fixed_question_id=FIXED_QUESTION_ID,
    ))
