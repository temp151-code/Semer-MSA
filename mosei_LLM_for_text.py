 # -*- coding: utf-8 -*-

import os
import re
import csv
import json
import time
import random
from typing import Dict, Set
from dotenv import load_dotenv

import requests

try:
    import numpy as np
except ImportError:
    np = None


# ============================================================
# 1. API Configuration
# ============================================================

# For privacy and security reasons, actual API credentials
# and service-specific connection settings are not included
# in the publicly released code.
#
# This section provides a configurable example.
# Users should store their API credentials and endpoint
# in a local .env file instead of hardcoding them.
#
# The .env file must NOT be committed to the repository.
# See .env.example for the required configuration fields.

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY", "").strip()
API_URL = os.getenv("LLM_API_URL", "").strip()

MODEL = "gpt-4o-mini"


# 固定生成随机种子
SEED = 42

# 0.0 通常更稳定；保留 0.2 可以产生轻微表达变化
TEMPERATURE = 0.2

TIMEOUT = 120
RETRY = 3

# 每条样本完成后暂停，减少限流风险
SLEEP_BETWEEN = 0.25


# ============================================================
# 2. 输入输出路径
# ============================================================

INPUT_CSV = "/root/autodl-tmp/MOSEI-label.csv"

# 文件名包含 seed，避免以后使用其他 seed 时混合已有结果
OUTPUT_JSONL = (
    f"/root/autodl-tmp/"
    f"MOSEI_text_semantic_explanations_seed{SEED}.jsonl"
)


# ============================================================
# 3. 固定本地随机种子
# ============================================================

def set_global_seed(seed: int) -> None:
    """
    固定本地 Python 和 NumPy 的随机状态。

    对远程大模型生成真正起主要作用的是：
        payload["seed"] = seed

    本地随机种子主要用于保证脚本自身涉及的随机操作可复现。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    print("=" * 70)
    print(f"Global seed: {seed}")
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print("=" * 70)


# ============================================================
# 4. 文本清理
# ============================================================

def clean_text(value) -> str:
    """
    将 CSV 中的字段安全转换为字符串。

    处理：
    - None
    - 普通字符串
    - 数字
    - 多余首尾空格
    """
    if value is None:
        return ""

    return str(value).strip()


# ============================================================
# 5. JSON 提取
# ============================================================

def extract_json_from_text(text: str) -> Dict:
    """
    尽量从模型输出中提取 JSON。

    理想输出：

    {
        "semantic_explanation": "..."
    }

    即使模型在 JSON 前后添加解释，也会尝试提取花括号内容。
    """
    text = clean_text(text)

    if not text:
        return {
            "semantic_explanation": ""
        }

    # 情况 1：整个输出就是标准 JSON
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    # 情况 2：JSON 外面带了 Markdown 代码块
    code_block_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1))

            if isinstance(parsed, dict):
                return parsed

        except Exception:
            pass

    # 情况 3：输出前后带其他文字
    json_match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if json_match:
        try:
            parsed = json.loads(json_match.group(0))

            if isinstance(parsed, dict):
                return parsed

        except Exception:
            pass

    # 最后兜底：把模型原始输出当成语义解释
    return {
        "semantic_explanation": text
    }


# ============================================================
# 6. API 请求
# ============================================================

def post_chat(payload: Dict) -> Dict:
    """
    Example API request function for an OpenAI-compatible
    chat completion endpoint.

    This is a configurable reference implementation.
    Users should adapt the endpoint, authentication method,
    request parameters, and response handling as needed
    for their own API provider and runtime environment.

    Actual API credentials and service-specific settings
    are intentionally excluded from the public code.
    """

    if not API_KEY or not API_URL:
        raise RuntimeError(
            "Please configure LLM_API_KEY and LLM_API_URL "
            "in your local .env file or environment."
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=TIMEOUT,
        )
        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            f"API request failed: {type(error).__name__}"
        ) from None

    result = response.json()

    if not isinstance(result, dict):
        raise ValueError(
            f"API response is not a dictionary: {type(result)}"
        )

    return result
# ============================================================
# 7. 生成语义解释
# ============================================================

def call_gpt_for_semantic_explanation(
    utterance: str
) -> Dict[str, str]:
    """
    为一条 MOSEI utterance 生成英文语义解释。

    只把 utterance 文本发送给 GPT。

    不发送：
    - label
    - annotation
    - mode
    - label_by

    因此 GPT 不会直接看到 MOSEI 真实标签。
    """
    utterance = clean_text(utterance)

    if not utterance:
        return {
            "semantic_explanation": ""
        }

    prompt = f"""
You are a semantic analysis assistant.

Given one spoken utterance from a video, write a concise English semantic explanation.

Your explanation should:

1. Explain what the speaker is communicating, implying, evaluating, or attempting to achieve.
2. Describe the speaker's stance, intention, attitude, or judgment only when it is supported by the utterance.
3. Identify semantic or pragmatic cues that may help a downstream model understand the utterance.
4. Add useful semantic interpretation rather than merely repeating the original sentence.
5. Do not invent speakers, events, relationships, emotions, or conversational context that are not supported by the utterance.
6. Do not mention any dataset, annotation, sentiment score, emotion label, ground-truth label, or classification category.
7. Do not provide a final label such as positive, negative, neutral, happy, sad, or angry.
8. Keep the explanation concise, normally one or two sentences.
9. Write the explanation entirely in English.

Return ONLY one valid JSON object using exactly this structure:

{{
  "semantic_explanation": "..."
}}

Utterance:
{json.dumps(utterance, ensure_ascii=False)}
""".strip()

    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,

        # 远程生成随机种子
        "seed": SEED,

        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
    }

    last_error = None

    for attempt in range(1, RETRY + 1):
        try:
            response_json = post_chat(payload)

            choices = response_json.get("choices", [])

            if not choices:
                raise ValueError(
                    "API response contains no choices."
                )

            message = choices[0].get("message", {})

            content = clean_text(
                message.get("content", "")
            )

            if not content:
                raise ValueError(
                    "The model returned empty content."
                )

            parsed_data = extract_json_from_text(content)

            semantic_explanation = clean_text(
                parsed_data.get(
                    "semantic_explanation",
                    ""
                )
            )

            if not semantic_explanation:
                raise ValueError(
                    "semantic_explanation is empty."
                )

            return {
                "semantic_explanation": semantic_explanation
            }

        except Exception as error:
            last_error = error

            print(
                f"  [WARN] attempt "
                f"{attempt}/{RETRY} failed: {error}"
            )

            if attempt < RETRY:
                wait_seconds = 1.0 + (attempt - 1) * 0.8

                print(
                    f"  [WAIT] sleep {wait_seconds:.1f} seconds"
                )

                time.sleep(wait_seconds)

    print(
        "  [ERROR] all retries failed; "
        "writing an empty semantic explanation."
    )
    print("  [ERROR] last error:", last_error)

    return {
        "semantic_explanation": ""
    }


# ============================================================
# 8. 样本唯一键
# ============================================================

def make_sample_key(
    video_id: str,
    clip_id: str
) -> str:
    """
    使用 video_id 和 clip_id 构造 MOSEI 样本唯一键。

    例如：
        video_id = "-3g5vACwYnA"
        clip_id = "10"

    得到：
        "-3g5vACwYnA|||10"
    """
    return f"{video_id}|||{clip_id}"


# ============================================================
# 9. 读取已经完成的结果
# ============================================================

def load_completed_keys(
    output_jsonl: str
) -> Set[str]:
    """
    从已有 JSONL 中读取已经完成的 video_id + clip_id。

    这样即使：
    - 程序中途停止；
    - CSV 顺序发生变化；
    - 某些行被跳过；

    仍然可以根据样本唯一键断点续跑。
    """
    completed_keys: Set[str] = set()

    if not os.path.exists(output_jsonl):
        print("No existing output file. Start from the beginning.")

        return completed_keys

    print("Loading existing output:", output_jsonl)

    with open(
        output_jsonl,
        "r",
        encoding="utf-8",
    ) as input_file:

        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)

                video_id = clean_text(
                    item.get("video_id", "")
                )

                clip_id = clean_text(
                    item.get("clip_id", "")
                )

                if not video_id or not clip_id:
                    print(
                        f"[WARN] output line {line_number} "
                        "has no video_id or clip_id."
                    )

                    continue

                key = make_sample_key(
                    video_id=video_id,
                    clip_id=clip_id,
                )

                completed_keys.add(key)

            except Exception as error:
                print(
                    f"[WARN] cannot parse output line "
                    f"{line_number}: {error}"
                )

    return completed_keys


# ============================================================
# 10. 检查 CSV 字段
# ============================================================

def validate_csv_columns(
    fieldnames
) -> None:
    """
    检查 MOSEI CSV 是否至少包含必要字段。
    """
    if fieldnames is None:
        raise RuntimeError(
            "CSV does not contain a header."
        )

    required_columns = {
        "video_id",
        "clip_id",
        "text",
    }

    current_columns = set(fieldnames)

    missing_columns = (
        required_columns - current_columns
    )

    if missing_columns:
        raise RuntimeError(
            "CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
            + "\nCurrent columns: "
            + ", ".join(fieldnames)
        )


# ============================================================
# 11. 处理整个 MOSEI CSV
# ============================================================

def annotate_mosei_csv_to_jsonl(
    input_csv: str,
    output_jsonl: str,
) -> None:
    """
    读取 MOSEI-label.csv，并为每个 utterance 生成：

        semantic_explanation

    预期输入字段：

        video_id
        clip_id
        text
        label
        annotation
        mode
        label_by

    输出 JSONL 字段：

        row_id
        video_id
        clip_id
        text
        label
        annotation
        mode
        label_by
        generation_model
        generation_seed
        generation_temperature
        semantic_explanation
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(
            f"Input CSV does not exist:\n{input_csv}"
        )

    output_directory = os.path.dirname(
        output_jsonl
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # 读取 CSV
    # --------------------------------------------------------

    with open(
        input_csv,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_file:

        reader = csv.DictReader(input_file)

        validate_csv_columns(reader.fieldnames)

        rows = list(reader)

    if not rows:
        raise RuntimeError(
            f"No data rows were found in:\n{input_csv}"
        )

    print("\nCSV columns:")
    print(reader.fieldnames)

    print("\nTotal CSV rows:", len(rows))

    # --------------------------------------------------------
    # 读取断点
    # --------------------------------------------------------

    completed_keys = load_completed_keys(
        output_jsonl
    )

    print("Already completed:", len(completed_keys))

    # 这里是估算值。
    # 如果 CSV 中存在重复 video_id + clip_id，remaining 可能不完全准确。
    estimated_remaining = max(
        0,
        len(rows) - len(completed_keys),
    )

    print("Estimated remaining:", estimated_remaining)

    # --------------------------------------------------------
    # 运行统计
    # --------------------------------------------------------

    new_success_count = 0
    new_empty_count = 0
    skipped_count = 0
    empty_text_count = 0
    invalid_id_count = 0

    # 使用追加模式，便于断点续跑
    with open(
        output_jsonl,
        "a",
        encoding="utf-8",
    ) as output_file:

        for row_id, row in enumerate(rows):
            video_id = clean_text(
                row.get("video_id", "")
            )

            clip_id = clean_text(
                row.get("clip_id", "")
            )

            text = clean_text(
                row.get("text", "")
            )

            label = clean_text(
                row.get("label", "")
            )

            annotation = clean_text(
                row.get("annotation", "")
            )

            mode = clean_text(
                row.get("mode", "")
            )

            label_by = clean_text(
                row.get("label_by", "")
            )

            # ------------------------------------------------
            # 检查样本 ID
            # ------------------------------------------------

            if not video_id or not clip_id:
                invalid_id_count += 1

                print("\n" + "=" * 70)
                print(f"[WARN] row {row_id} has invalid ID")
                print("video_id:", repr(video_id))
                print("clip_id:", repr(clip_id))
                print("Skip this row.")

                continue

            sample_key = make_sample_key(
                video_id=video_id,
                clip_id=clip_id,
            )

            # ------------------------------------------------
            # 已经完成则跳过
            # ------------------------------------------------

            if sample_key in completed_keys:
                skipped_count += 1
                continue

            print("\n" + "=" * 70)
            print(
                f"Row: {row_id + 1}/{len(rows)}"
            )
            print("video_id:", video_id)
            print("clip_id:", clip_id)
            print("mode:", mode)
            print("label:", label)
            print("annotation:", annotation)
            print("text:", text)

            # ------------------------------------------------
            # 生成语义解释
            # ------------------------------------------------

            if text:
                result = (
                    call_gpt_for_semantic_explanation(
                        utterance=text
                    )
                )

                semantic_explanation = clean_text(
                    result.get(
                        "semantic_explanation",
                        ""
                    )
                )

            else:
                semantic_explanation = ""
                empty_text_count += 1

                print(
                    "[WARN] text is empty; "
                    "no API request was sent."
                )

            # ------------------------------------------------
            # 组织输出
            # ------------------------------------------------

            output_object = {
                "row_id": row_id,
                "video_id": video_id,
                "clip_id": clip_id,
                "text": text,

                # 原始 MOSEI 标签和元信息
                # 这些内容不会发送给 GPT
                "label": label,
                "annotation": annotation,
                "mode": mode,
                "label_by": label_by,

                # 生成配置，方便实验追溯
                "generation_model": MODEL,
                "generation_seed": SEED,
                "generation_temperature": TEMPERATURE,

                # 唯一新增的 GPT 语义标注
                "semantic_explanation": semantic_explanation,
            }

            # ------------------------------------------------
            # 立即写入磁盘
            # ------------------------------------------------

            output_file.write(
                json.dumps(
                    output_object,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

            # 写入成功后才加入 completed_keys
            completed_keys.add(sample_key)

            # ------------------------------------------------
            # 更新统计
            # ------------------------------------------------

            if semantic_explanation:
                new_success_count += 1

                print(
                    "semantic_explanation:",
                    semantic_explanation,
                )

            else:
                new_empty_count += 1

                print(
                    "semantic_explanation: <EMPTY>"
                )

            print(
                "\n[Progress] "
                f"completed_keys={len(completed_keys)}/"
                f"{len(rows)} | "
                f"new_success={new_success_count} | "
                f"new_empty={new_empty_count} | "
                f"skipped={skipped_count}"
            )

            time.sleep(SLEEP_BETWEEN)

    # --------------------------------------------------------
    # 完成报告
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MOSEI semantic annotation finished")
    print("=" * 70)

    print("Input CSV:", input_csv)
    print("Output JSONL:", output_jsonl)

    print("Model:", MODEL)
    print("Seed:", SEED)
    print("Temperature:", TEMPERATURE)

    print("Total CSV rows:", len(rows))
    print("New successful rows:", new_success_count)
    print("New empty-result rows:", new_empty_count)
    print("Empty-text rows:", empty_text_count)
    print("Skipped existing rows:", skipped_count)
    print("Invalid-ID rows:", invalid_id_count)


# ============================================================
# 12. 主程序
# ============================================================

if __name__ == "__main__":
    set_global_seed(SEED)

    annotate_mosei_csv_to_jsonl(
        input_csv=INPUT_CSV,
        output_jsonl=OUTPUT_JSONL,
    ) 