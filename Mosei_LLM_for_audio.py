# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import requests

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ============================================================
# 1. 配置区
# ============================================================

CSV_PATH = (
    "/root/autodl-tmp/"
    "MOSEI_audio_features_binned.csv"
)

OUT_DIR = "/root/autodl-tmp"

OUT_JSONL = os.path.join(
    OUT_DIR,
    "MOSEI_audio_llm_seed42.jsonl",
)

LOG_PATH = os.path.join(
    OUT_DIR,
    "MOSEI_audio_llm_seed42.log",
)

# ============================================================
# API Configuration
# ============================================================

# For privacy and security reasons, actual API credentials
# and service-specific connection settings are not included
# in the publicly released code.
#
# This section provides a configurable example.
# Users should store their own API credentials and endpoint
# in a local .env file rather than hardcoding them here.
#
# The .env file must NOT be committed to the repository.
# See .env.example for the required configuration fields.



load_dotenv()

API_KEY = os.getenv("LLM_API_KEY", "").strip()
API_URL = os.getenv("LLM_API_URL", "").strip()

if not API_KEY or not API_URL:
    raise RuntimeError(
        "Please configure LLM_API_KEY and LLM_API_URL "
        "in your local .env file."
    )



MODEL = "gpt-4o-mini"

TEMPERATURE = 0.2
MAX_TOKENS = 320
RETRY = 3
TIMEOUT = 120

# 两次成功请求之间的基础等待时间
SLEEP_BETWEEN = 0.25


# ============================================================
# 2. 随机种子
# ============================================================

RANDOM_SEED = 42

# 尝试把 seed 传给远程模型。
# 如果当前接口不支持 seed，
# 程序会自动退回到不传 seed 的方式。
API_SEED = 42

# test：随机测试 TEST_N 条，不写入正式结果
# full：正式全量运行
RUN_MODE = "full"

TEST_N = 20


# ============================================================
# 3. 限流重试
# ============================================================

BACKOFF_BASE = 2.0
BACKOFF_CAP = 30.0
JITTER = 0.4


# ============================================================
# 4. Prompt
# ============================================================

SYSTEM_PROMPT = """You are an acoustic prosody analyst.

Hard rules:
- Use ONLY the provided features and relative bins.
- Do not invent numbers, measurements, speakers, contexts, or fields.
- If a field is not provided, you MUST NOT mention it.
- Do NOT infer or mention sentiment, emotion labels, intentions, or mental states.
- Do NOT use words such as anger, joy, sadness, fear, surprise, disgust, neutral, positive, or negative.
- Describe acoustic delivery only.
- Output STRICT JSON only, with no markdown and no extra text.

Output format:
{
  "claims": [
    {"claim": "..."}
  ],
  "style_tags": ["..."],
  "one_liner": "...",
  "evidence_pool": {
    "feature_name": "value"
  }
}

Constraints:
- If pitch numeric features are provided, output exactly 3 claims.
- If pitch numeric features are not provided, output exactly 2 claims.
- Each claim must contain no more than 16 words.
- Do not put numeric digits inside claims.
- evidence_pool must contain ONLY provided numeric features or bins actually used.
- Round numeric values to at most 3 decimal places.
- When using low, mid, or high descriptions, follow the provided bin fields.
- style_tags must contain 3 to 6 snake_case tags.
- one_liner must contain no more than 18 words.
- one_liner must be a complete sentence ending with a period.

Content planning:

If pitch numeric features are provided:
1. Claim 1: describe duration and speaking-rate characteristics.
2. Claim 2: describe loudness and loudness dynamics.
3. Claim 3: describe pitch level, variability, or trend.

If pitch numeric features are not provided:
1. Claim 1: describe duration and speaking-rate characteristics.
2. Claim 2: describe loudness, dynamics, or voicing.
- Do not mention pitch, intonation, rising, falling, or pitch variability.

Pause rule:
- Mention pauses only when pause_ratio is at least 0.23 or at most 0.19.
- Otherwise, do not describe pauses.

one_liner rule:
- Summarize at least two available aspects among duration, rate, loudness, dynamics, and voicing.
- Avoid generic openings such as "The analysis reveals" or "The acoustic features show".
"""


HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


# ============================================================
# 5. 全局状态
# ============================================================
# 预防seed报错
# 如果第一次遇到 seed 参数报错，后续请求自动不再发送 seed。
API_SEED_SUPPORTED = True


# ============================================================
# 6. 日志
# ============================================================

def log(message: str) -> None:
    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    line = f"[{timestamp}] {message}"

    print(
        line,
        flush=True,
    )

    try:
        os.makedirs(
            os.path.dirname(LOG_PATH),
            exist_ok=True,
        )

        with open(
            LOG_PATH,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                line + "\n"
            )

    except Exception:
        pass


# ============================================================
# 7. 基础清理函数
# ============================================================

def is_missing(value) -> bool:
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    text = str(value).strip().lower()

    return text in {
        "",
        "nan",
        "none",
        "null",
        "missing",
    }


def clean_id(value):
    """
    避免 clip_id 被读成 1.0、2.0。
    """
    if is_missing(value):
        return None

    value = str(value).strip()

    if re.fullmatch(
        r"-?\d+\.0",
        value,
    ):
        value = value[:-2]

    return value


def clean_optional_text(value):
    if is_missing(value):
        return None

    return str(value).strip()


def r3(value):
    """
    数值统一保留最多三位小数。
    缺失或异常值返回 None。
    """
    if is_missing(value):
        return None

    try:
        number = float(value)

        if not np.isfinite(number):
            return None

        return float(
            np.round(
                number,
                3,
            )
        )

    except Exception:
        return None


# ============================================================
# 8. 读取 MOSEI CSV
# ============================================================

def read_mosei_csv() -> pd.DataFrame:
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Input CSV does not exist:\n{CSV_PATH}"
        )

    dataframe = pd.read_csv(
        CSV_PATH,
        dtype={
            "video_id": str,
            "clip_id": str,
            "mode": str,
        },
        low_memory=False,
    )

    required_columns = {
        "video_id",
        "clip_id",
        "mode",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "MOSEI CSV is missing required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    dataframe["video_id"] = (
        dataframe["video_id"]
        .apply(clean_id)
    )

    dataframe["clip_id"] = (
        dataframe["clip_id"]
        .apply(clean_id)
    )

    dataframe["mode"] = (
        dataframe["mode"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return dataframe


# ============================================================
# 9. API 请求
# ============================================================

def call_llm(
    system_prompt: str,
    user_prompt: str,
):
    global API_SEED_SUPPORTED

    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    }

    if (
        API_SEED is not None
        and API_SEED_SUPPORTED
    ):
        payload["seed"] = API_SEED

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=TIMEOUT,
    )

    # 某些兼容接口可能不支持 seed
    if (
        response.status_code == 400
        and "seed" in payload
        and "seed" in (
            response.text or ""
        ).lower()
    ):
        API_SEED_SUPPORTED = False

        log(
            "[WARN] API rejected the seed parameter; "
            "retrying without API seed."
        )

        payload.pop(
            "seed",
            None,
        )

        response = requests.post(
            API_URL,
            headers=HEADERS,
            json=payload,
            timeout=TIMEOUT,
        )

    return response


# ============================================================
# 10. JSON 解析
# ============================================================

def safe_json_load(text: str):
    """
    严格读取 JSON。

    若模型输出 markdown 代码块，会触发重试，
    而不是静默清洗后接受。
    """
    return json.loads(
        text.strip()
    )


# ============================================================
# 11. MOSEI 唯一键
# ============================================================

def record_key(row) -> str:
    video_id = clean_id(
        row.get(
            "video_id",
            None,
        )
    )

    clip_id = clean_id(
        row.get(
            "clip_id",
            None,
        )
    )

    if not video_id:
        raise ValueError(
            "video_id is missing"
        )

    if not clip_id:
        raise ValueError(
            "clip_id is missing"
        )

    return (
        f"{video_id}"
        f"|||"
        f"{clip_id}"
    )


# ============================================================
# 12. 读取已经完成的样本
# ============================================================

def load_done_keys(
    output_jsonl: str,
):
    """
    只有以下记录会被视为完成：

    1. status == ok
    2. status == skipped
    3. 兼容旧格式：存在 llm 且没有 error

    API 请求失败的 error 记录不会被视为完成，
    因而重新运行脚本时会继续尝试。
    """
    done = set()

    try:
        with open(
            output_jsonl,
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                try:
                    obj = json.loads(
                        line
                    )

                    key = obj.get(
                        "_key"
                    )

                    if not key:
                        continue

                    status = obj.get(
                        "status"
                    )

                    if status in {
                        "ok",
                        "skipped",
                    }:
                        done.add(
                            key
                        )

                    elif (
                        isinstance(
                            obj.get("llm"),
                            dict,
                        )
                        and not obj.get(
                            "error"
                        )
                    ):
                        done.add(
                            key
                        )

                except Exception:
                    continue

    except FileNotFoundError:
        pass

    return done


# ============================================================
# 13. 构建输入 Prompt
# ============================================================

def build_user_prompt(
    row: dict,
):
    """
    注意：

    1. 不向 LLM 提供 label。
    2. 不向 LLM 提供 annotation。
    3. 不向 LLM 提供文本内容。
    4. 缺失字段不会进入 Prompt。
    """

    numeric = {
        "duration_sec": r3(
            row.get("duration_sec")
        ),
        "rms_mean": r3(
            row.get("rms_mean")
        ),
        "rms_p95": r3(
            row.get("rms_p95")
        ),
        "rms_range_p95_p05": r3(
            row.get(
                "rms_range_p95_p05"
            )
        ),
        "voiced_ratio": r3(
            row.get("voiced_ratio")
        ),
        "pause_ratio": r3(
            row.get("pause_ratio")
        ),
        "rate_proxy_peaks_per_sec": r3(
            row.get(
                "rate_proxy_peaks_per_sec"
            )
        ),
    }

    f0_mean = r3(
        row.get("f0_mean_hz")
    )

    f0_std = r3(
        row.get("f0_std_hz")
    )

    f0_slope = r3(
        row.get(
            "f0_slope_hz_per_sec"
        )
    )
    # 只有存在非零有效平均基频时，
    # 才视为有可用 pitch numeric。
    pitch_present = (
        f0_mean is not None
        and f0_mean != 0.0
    )

    if pitch_present:
        numeric["f0_mean_hz"] = (
            f0_mean
        )

        # 0 也是合法数值，例如音高变化非常小，
        # 因此这里只排除 None。
        if f0_std is not None:
            numeric["f0_std_hz"] = (
                f0_std
            )

        if f0_slope is not None:
            numeric[
                "f0_slope_hz_per_sec"
            ] = f0_slope

    bins = {
        "loudness_bin": (
            clean_optional_text(
                row.get("loudness_bin")
            )
        ),
        "loudness_dynamics_bin": (
            clean_optional_text(
                row.get(
                    "loudness_dynamics_bin"
                )
            )
        ),
        "pitch_variability_bin": (
            clean_optional_text(
                row.get(
                    "pitch_variability_bin"
                )
            )
        ),
        "voicing_bin": (
            clean_optional_text(
                row.get("voicing_bin")
            )
        ),
        "rate_bin": (
            clean_optional_text(
                row.get("rate_bin")
            )
        ),
        "pitch_trend": (
            clean_optional_text(
                row.get("pitch_trend")
            )
        ),
        "duration_bin": (
            clean_optional_text(
                row.get("duration_bin")
            )
        ),
        "noise_risk": (
            clean_optional_text(
                row.get("noise_risk")
            )
        ),
        "speech_reliable": (
            clean_optional_text(
                row.get(
                    "speech_reliable"
                )
            )
        ),
    }

    # 无有效 pitch numeric 时，
    # 不向模型提供音高离散字段。
    if not pitch_present:
        bins.pop(
            "pitch_variability_bin",
            None,
        )

        bins.pop(
            "pitch_trend",
            None,
        )

    numeric = {
        key: value
        for key, value
        in numeric.items()
        if value is not None
    }

    bins = {
        key: value
        for key, value
        in bins.items()
        if value is not None
    }

    lines = [
        (
            "Produce the required JSON strictly "
            "following the format and constraints."
        ),
        "",
        "Provided numeric features:",
    ]

    for key, value in numeric.items():
        lines.append(
            f"{key}={value}"
        )

    lines.extend([
        "",
        "Provided relative bins:",
    ])

    for key, value in bins.items():
        lines.append(
            f"{key}={value}"
        )

    return (
        "\n".join(lines),
        pitch_present,
        numeric,
        bins,
    )


# ============================================================
# 14. 输入是否足够
# ============================================================

def has_sufficient_features(
    numeric: dict,
    bins: dict,
) -> bool:
    """
    one_liner 至少需要概括两个方面。

    如果音频特征提取失败，所有字段都为空，
    则不浪费 API 请求。
    """
    summary_bin_keys = {
        "duration_bin",
        "loudness_bin",
        "loudness_dynamics_bin",
        "voicing_bin",
        "rate_bin",
        "noise_risk",
    }

    available_summary_bins = (
        summary_bin_keys
        & set(bins.keys())
    )

    if len(available_summary_bins) >= 2:
        return True

    numeric_summary_keys = {
        "duration_sec",
        "rms_mean",
        "rms_range_p95_p05",
        "voiced_ratio",
        "rate_proxy_peaks_per_sec",
    }

    available_numeric = (
        numeric_summary_keys
        & set(numeric.keys())
    )

    return len(available_numeric) >= 2


# ============================================================
# 15. 输出校验
# ============================================================

BANNED_EMOTION_PATTERN = re.compile(
    r"\b("
    r"anger|angry|joy|joyful|"
    r"sadness|sad|fear|fearful|"
    r"surprise|surprised|"
    r"disgust|disgusted|"
    r"neutral|positive|negative|"
    r"emotion|emotional|"
    r"sentiment"
    r")\b",
    flags=re.IGNORECASE,
)


def validate_llm_json(
    parsed,
    pitch_present: bool,
    numeric: dict,
    bins: dict,
):
    if not isinstance(
        parsed,
        dict,
    ):
        return (
            False,
            "output is not a JSON object",
        )

    required_keys = {
        "claims",
        "style_tags",
        "one_liner",
        "evidence_pool",
    }

    missing_keys = (
        required_keys
        - set(parsed.keys())
    )

    if missing_keys:
        return (
            False,
            "missing top-level keys: "
            + ", ".join(
                sorted(missing_keys)
            ),
        )

    # --------------------------------------------------------
    # Claims
    # --------------------------------------------------------

    expected_claims = (
        3
        if pitch_present
        else 2
    )

    claims = parsed["claims"]

    if (
        not isinstance(claims, list)
        or len(claims)
        != expected_claims
    ):
        return (
            False,
            (
                "claims count mismatch: "
                f"expected {expected_claims}, "
                f"got "
                f"{len(claims) if isinstance(claims, list) else 'invalid'}"
            ),
        )

    claim_texts = []

    for claim_item in claims:
        if (
            not isinstance(
                claim_item,
                dict,
            )
            or "claim"
            not in claim_item
        ):
            return (
                False,
                "invalid claim item",
            )

        claim_text = str(
            claim_item["claim"]
        ).strip()

        if not claim_text:
            return (
                False,
                "empty claim",
            )

        if any(
            character.isdigit()
            for character in claim_text
        ):
            return (
                False,
                "claim contains numeric digits",
            )

        if len(
            claim_text.split()
        ) > 16:
            return (
                False,
                "claim exceeds 16 words",
            )

        claim_texts.append(
            claim_text
        )

    # --------------------------------------------------------
    # One-liner
    # --------------------------------------------------------

    one_liner = parsed[
        "one_liner"
    ]

    if (
        not isinstance(
            one_liner,
            str,
        )
        or not one_liner.strip()
    ):
        return (
            False,
            "one_liner is empty",
        )

    one_liner = (
        one_liner.strip()
    )

    if not one_liner.endswith(
        "."
    ):
        return (
            False,
            "one_liner must end with a period",
        )

    if len(
        one_liner.split()
    ) > 18:
        return (
            False,
            "one_liner exceeds 18 words",
        )

    # --------------------------------------------------------
    # 禁止情绪推断
    # --------------------------------------------------------

    combined_text = " ".join(
        claim_texts
        + [one_liner]
    )

    if BANNED_EMOTION_PATTERN.search(
        combined_text
    ):
        return (
            False,
            "output contains banned emotion or sentiment wording",
        )

    # --------------------------------------------------------
    # 无 pitch 时禁止提及音高
    # --------------------------------------------------------

    if not pitch_present:
        lower_text = (
            combined_text.lower()
        )

        banned_pitch_words = [
            "pitch",
            "f0",
            "intonation",
            "rising",
            "falling",
            "melodic",
        ]

        for word in banned_pitch_words:
            if word in lower_text:
                return (
                    False,
                    (
                        "no-pitch case mentions "
                        f"pitch-related word: {word}"
                    ),
                )

    # --------------------------------------------------------
    # Style tags
    # --------------------------------------------------------

    style_tags = parsed[
        "style_tags"
    ]

    if (
        not isinstance(
            style_tags,
            list,
        )
        or not (
            3
            <= len(style_tags)
            <= 6
        )
    ):
        return (
            False,
            "style_tags count must be between 3 and 6",
        )

    for tag in style_tags:
        if not isinstance(
            tag,
            str,
        ):
            return (
                False,
                "style tag is not a string",
            )

        if not re.fullmatch(
            r"[a-z0-9]+(?:_[a-z0-9]+)*",
            tag.strip(),
        ):
            return (
                False,
                (
                    "style tag is not "
                    f"snake_case: {tag}"
                ),
            )

        if BANNED_EMOTION_PATTERN.search(
            tag
        ):
            return (
                False,
                "style tag contains emotion wording",
            )

    # --------------------------------------------------------
    # Evidence pool
    # --------------------------------------------------------

    evidence_pool = parsed[
        "evidence_pool"
    ]

    if not isinstance(
        evidence_pool,
        dict,
    ):
        return (
            False,
            "evidence_pool is not a JSON object",
        )

    allowed_evidence_keys = (
        set(numeric.keys())
        | set(bins.keys())
    )

    extra_evidence_keys = (
        set(evidence_pool.keys())
        - allowed_evidence_keys
    )

    if extra_evidence_keys:
        return (
            False,
            (
                "evidence_pool contains "
                "unprovided keys: "
                + ", ".join(
                    sorted(
                        extra_evidence_keys
                    )
                )
            ),
        )

    return (
        True,
        "ok",
    )


# ============================================================
# 16. 退避等待
# ============================================================

def sleep_backoff(
    attempt: int,
    reason: str,
):
    wait_base = min(
        BACKOFF_CAP,
        BACKOFF_BASE
        * (
            2
            ** (
                attempt - 1
            )
        ),
    )

    jitter = (
        wait_base
        * JITTER
        * (
            random.random()
            * 2
            - 1
        )
    )

    wait_seconds = max(
        0.5,
        wait_base + jitter,
    )

    log(
        f"Backoff {wait_seconds:.2f}s "
        f"due to {reason}"
    )

    time.sleep(
        wait_seconds
    )


# ============================================================
# 17. 请求单条样本
# ============================================================

def request_one_sample(
    user_prompt: str,
    pitch_present: bool,
    numeric: dict,
    bins: dict,
):
    last_error = None

    for attempt in range(
        1,
        RETRY + 1,
    ):
        try:
            response = call_llm(
                SYSTEM_PROMPT,
                user_prompt,
            )

            status_code = (
                response.status_code
            )

            if status_code == 200:
                response_json = (
                    response.json()
                )

                output_text = (
                    response_json[
                        "choices"
                    ][0][
                        "message"
                    ][
                        "content"
                    ]
                )

                parsed = safe_json_load(
                    output_text
                )

                valid, message = (
                    validate_llm_json(
                        parsed=parsed,
                        pitch_present=(
                            pitch_present
                        ),
                        numeric=numeric,
                        bins=bins,
                    )
                )

                if not valid:
                    raise ValueError(
                        message
                    )

                return (
                    parsed,
                    None,
                )

            response_snippet = (
                response.text or ""
            )[:300].replace(
                "\n",
                "\\n",
            )

            last_error = (
                f"HTTP {status_code}: "
                f"{response_snippet}"
            )

            if (
                status_code == 429
                or 500
                <= status_code
                < 600
            ):
                sleep_backoff(
                    attempt,
                    f"HTTP {status_code}",
                )

            else:
                time.sleep(
                    0.8 * attempt
                )

        except Exception as error:
            last_error = (
                f"{type(error).__name__}: "
                f"{str(error)}"
            )

            time.sleep(
                0.8 * attempt
            )

    return (
        None,
        last_error,
    )


# ============================================================
# 18. 测试模式
# ============================================================

def run_test_print():
    dataframe = (
        read_mosei_csv()
    )

    log(
        f"[TEST] CSV={CSV_PATH}"
    )

    log(
        f"[TEST] total rows={len(dataframe)}"
    )

    log(
        "[TEST] split distribution="
        + str(
            dataframe["mode"]
            .value_counts()
            .to_dict()
        )
    )

    sample_dataframe = (
        dataframe.sample(
            n=min(
                TEST_N,
                len(dataframe),
            ),
            random_state=(
                RANDOM_SEED
            ),
        )
    )

    for _, row in (
        sample_dataframe.iterrows()
    ):
        key = record_key(
            row
        )

        (
            user_prompt,
            pitch_present,
            numeric,
            bins,
        ) = build_user_prompt(
            row.to_dict()
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "Key:",
            key,
        )

        print(
            "video_id:",
            row.get("video_id"),
        )

        print(
            "clip_id:",
            row.get("clip_id"),
        )

        print(
            "mode:",
            row.get("mode"),
        )

        print(
            "pitch_present:",
            pitch_present,
        )

        print(
            "\nProvided numeric:"
        )

        print(
            json.dumps(
                numeric,
                ensure_ascii=False,
                indent=2,
            )
        )

        print(
            "\nProvided bins:"
        )

        print(
            json.dumps(
                bins,
                ensure_ascii=False,
                indent=2,
            )
        )

        if not has_sufficient_features(
            numeric,
            bins,
        ):
            print(
                "\nSKIPPED: insufficient acoustic features"
            )

            continue

        parsed, error = (
            request_one_sample(
                user_prompt=(
                    user_prompt
                ),
                pitch_present=(
                    pitch_present
                ),
                numeric=numeric,
                bins=bins,
            )
        )

        if error:
            print(
                "\nFAILED:",
                error,
            )

        else:
            print(
                "\nLLM output:"
            )

            print(
                json.dumps(
                    parsed,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        time.sleep(
            SLEEP_BETWEEN
        )


# ============================================================
# 19. 全量运行
# ============================================================

def run_full_to_jsonl():
    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )

    dataframe = (
        read_mosei_csv()
    )

    log(
        f"[FULL] CSV={CSV_PATH}"
    )

    log(
        f"[FULL] total rows={len(dataframe)}"
    )

    log(
        "[FULL] split distribution="
        + str(
            dataframe["mode"]
            .value_counts()
            .to_dict()
        )
    )

    if len(dataframe) > 0:
        first_row = (
            dataframe.iloc[0]
        )

        log(
            "[FULL] sample key="
            f"{record_key(first_row)}"
        )

    done_keys = load_done_keys(
        OUT_JSONL
    )

    log(
        f"[FULL] OUT_JSONL={OUT_JSONL}"
    )

    log(
        f"[FULL] already completed="
        f"{len(done_keys)}"
    )

    total_rows = len(
        dataframe
    )

    success_count = 0
    error_count = 0
    skipped_count = 0
    existing_count = 0
    visited_count = 0

    with open(
        OUT_JSONL,
        "a",
        encoding="utf-8",
    ) as output_file:

        for row_index, row in (
            dataframe.iterrows()
        ):
            try:
                key = record_key(
                    row
                )

            except Exception as error:
                log(
                    f"[ERR] row={row_index} "
                    f"invalid key: {error}"
                )

                continue

            if key in done_keys:
                existing_count += 1
                continue

            visited_count += 1

            (
                user_prompt,
                pitch_present,
                numeric,
                bins,
            ) = build_user_prompt(
                row.to_dict()
            )

            base_output = {
                "_key": key,
                "row_id": int(
                    row_index
                ),
                "video_id": clean_id(
                    row.get("video_id")
                ),
                "clip_id": clean_id(
                    row.get("clip_id")
                ),
                "mode": clean_optional_text(
                    row.get("mode")
                ),
                "mp3_path": (
                    clean_optional_text(
                        row.get("mp3_path")
                    )
                ),
                "pitch_present": (
                    pitch_present
                ),
                "provided": {
                    "numeric": numeric,
                    "bins": bins,
                },
                "generation": {
                    "model": MODEL,
                    "temperature": (
                        TEMPERATURE
                    ),
                    "requested_seed": (
                        API_SEED
                    ),
                    "local_seed": (
                        RANDOM_SEED
                    ),
                },
            }

            # label 和 annotation 故意不写入 Prompt。
            # 这里也不复制到输出，之后通过
            # video_id + clip_id 回连原 CSV。

            if not has_sufficient_features(
                numeric,
                bins,
            ):
                output = {
                    **base_output,
                    "status": "skipped",
                    "error": (
                        "Insufficient usable "
                        "acoustic features"
                    ),
                }

                output_file.write(
                    json.dumps(
                        output,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()

                done_keys.add(
                    key
                )

                skipped_count += 1

                log(
                    f"[SKIP] {key} | "
                    "insufficient features"
                )

                continue

            parsed, error = (
                request_one_sample(
                    user_prompt=(
                        user_prompt
                    ),
                    pitch_present=(
                        pitch_present
                    ),
                    numeric=numeric,
                    bins=bins,
                )
            )

            if error is None:
                output = {
                    **base_output,
                    "status": "ok",
                    "llm": parsed,
                }

                output_file.write(
                    json.dumps(
                        output,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()

                done_keys.add(
                    key
                )

                success_count += 1

            else:
                output = {
                    **base_output,
                    "status": "error",
                    "error": error,
                }

                output_file.write(
                    json.dumps(
                        output,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()

                error_count += 1

                log(
                    f"[ERR] {key} | {error}"
                )

            if (
                visited_count % 200
                == 0
            ):
                log(
                    "[FULL] progress "
                    f"visited={visited_count}, "
                    f"success={success_count}, "
                    f"errors={error_count}, "
                    f"skipped={skipped_count}, "
                    f"existing={existing_count}, "
                    f"total={total_rows}"
                )

            time.sleep(
                SLEEP_BETWEEN
            )

    log(
        "[FULL] finished | "
        f"success={success_count}, "
        f"errors={error_count}, "
        f"skipped={skipped_count}, "
        f"existing={existing_count}"
    )

    log(
        f"[FULL] output -> {OUT_JSONL}"
    )


# ============================================================
# 20. 启动
# ============================================================

if __name__ == "__main__":
    if not API_KEY:
        raise RuntimeError(
            "LLM_API_KEY is empty. "
            "Please export it before running."
        )

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    if RUN_MODE == "test":
        run_test_print()

    elif RUN_MODE == "full":
        run_full_to_jsonl()

    else:
        raise ValueError(
            "RUN_MODE must be 'test' or 'full'."
        )