# -*- coding: utf-8 -*-

import os
import re
import json
import time
import base64
import random
from typing import Dict, List, Optional, Set, Tuple
from dotenv import load_dotenv

import requests

try:
    import numpy as np
except ImportError:
    np = None


# ============================================================
# 1. 路径配置
# ============================================================

ROOT_DIR = "/root/autodl-tmp/MOSEI_test/mosei_move/Raw"

OUT_JSONL = (
    "/root/autodl-tmp/"
    "MOSEI_visual_LLM_seed42.jsonl"
)

# ============================================================
# 2. API Configuration
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

# 固定随机种子
SEED = 42

# 低温度配合 seed，使生成更加稳定
TEMPERATURE = 0.2

TIMEOUT = 120
RETRY = 3

# 每处理一条后暂停，降低触发限流的概率
SLEEP_BETWEEN = 0.25

# 图像细节：
# "low"：成本较低
# "high"：更有利于观察细粒度面部特征
IMAGE_DETAIL = "high"

# None 或 -1：处理全部
# 1：只测试一条
TEST_N = -1


# ============================================================
# 3. 禁止出现的情绪和心理状态词
# ============================================================

EMOTION_BAN_PATTERN = re.compile(
    r"\b("
    r"positive|negative|neutral|"
    r"happy|happiness|joy|joyful|pleased|cheerful|"
    r"sad|sadness|unhappy|depressed|depression|upset|"
    r"angry|anger|furious|mad|irritated|irritation|"
    r"annoyed|annoyance|"
    r"afraid|fear|fearful|scared|terrified|"
    r"anxious|anxiety|nervous|tense|calm|relaxed|"
    r"excited|excitement|"
    r"surprised|surprise|shocked|astonished|amazed|"
    r"disgusted|disgust|repulsed|"
    r"confident|uncertain|hesitant|"
    r"emotion|emotional|feeling|feelings|mood|"
    r"intention|intentions|mental|psychological"
    r")\b",
    flags=re.IGNORECASE,
)


# ============================================================
# 4. 固定随机种子
# ============================================================

def set_global_seed(seed: int) -> None:
    """
    固定本地 Python 和 NumPy 随机状态。

    远程模型的生成稳定性主要依靠：
        payload["seed"] = seed

    但远程服务升级模型后，仍不能保证永久逐字一致。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)

    if np is not None:
        np.random.seed(seed)

    print("=" * 72)
    print("Global seed:", seed)
    print("Model:", MODEL)
    print("Temperature:", TEMPERATURE)
    print("Image detail:", IMAGE_DETAIL)
    print("=" * 72)


# ============================================================
# 5. 基础工具
# ============================================================

def clean_text(value) -> str:
    """
    安全地把字段转换成字符串。
    """
    if value is None:
        return ""

    return str(value).strip()


def ensure_output_directory(path: str) -> None:
    """
    确保输出目录存在。
    """
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )


def natural_sort_key(text: str):
    """
    自然排序。

    例如：
        2 排在 10 前面；
        0002.png 排在 0010.png 前面。
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    ]


def make_sample_key(
    video_id: str,
    clip_id: str,
) -> str:
    """
    构造 MOSEI 样本唯一键。
    """
    return f"{video_id}|||{clip_id}"


# ============================================================
# 6. 已完成样本读取
# ============================================================

def load_completed_keys(
    output_jsonl: str,
) -> Set[str]:
    """
    从已有 JSONL 中读取已经成功处理的样本。

    使用 video_id + clip_id 判断，不依赖文件行数。
    """
    completed_keys: Set[str] = set()

    if not os.path.exists(output_jsonl):
        print("No existing output file.")
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

                visual_oneline = clean_text(
                    item.get("visual_oneline", "")
                )

                # 只有成功生成了描述，才视为完成
                if video_id and clip_id and visual_oneline:
                    completed_keys.add(
                        make_sample_key(
                            video_id,
                            clip_id,
                        )
                    )

            except Exception as error:
                print(
                    f"[WARN] Cannot parse output line "
                    f"{line_number}: {error}"
                )

    return completed_keys


# ============================================================
# 7. 遍历 MOSEI 片段目录
# ============================================================

def iter_clip_directories(
    root_dir: str,
):
    """
    遍历：

    Raw/
    └── video_id/
        ├── clip_id/
        │   ├── 0000.png
        │   ├── 0001.png
        │   └── ...
        ├── clip_id.mp3
        └── ...

    只返回目录，不返回 mp3 文件。
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"ROOT_DIR does not exist:\n{root_dir}"
        )

    video_ids = sorted(
        os.listdir(root_dir),
        key=natural_sort_key,
    )

    for video_id in video_ids:
        video_path = os.path.join(
            root_dir,
            video_id,
        )

        if not os.path.isdir(video_path):
            continue

        clip_entries = sorted(
            os.listdir(video_path),
            key=natural_sort_key,
        )

        for clip_id in clip_entries:
            clip_path = os.path.join(
                video_path,
                clip_id,
            )

            # 自动忽略 3.mp3、4.mp3 等文件
            if not os.path.isdir(clip_path):
                continue

            yield (
                video_id,
                clip_id,
                video_path,
                clip_path,
            )


# ============================================================
# 8. 选择中间帧
# ============================================================

def list_image_frames(
    clip_path: str,
) -> List[str]:
    """
    返回一个 clip 目录中的全部图像帧。
    """
    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    )

    frames: List[str] = []

    for filename in os.listdir(clip_path):
        lower_name = filename.lower()

        if not lower_name.endswith(valid_extensions):
            continue

        # 避免未来目录中出现额外的人脸裁剪文件
        if "_faces" in lower_name:
            continue

        full_path = os.path.join(
            clip_path,
            filename,
        )

        if os.path.isfile(full_path):
            frames.append(filename)

    frames.sort(key=natural_sort_key)

    return frames


def pick_middle_frame(
    clip_path: str,
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[int],
    int,
]:
    """
    从一个 clip 中选择中间帧。

    返回：
        frame_path
        frame_id
        middle_index
        total_frames

    偶数帧时选择靠后的中间帧。

    例如 8 帧：
        index = 4
    """
    frames = list_image_frames(clip_path)

    total_frames = len(frames)

    if total_frames == 0:
        return None, None, None, 0

    middle_index = total_frames // 2
    middle_filename = frames[middle_index]

    frame_path = os.path.join(
        clip_path,
        middle_filename,
    )

    frame_id = os.path.splitext(
        middle_filename
    )[0]

    return (
        frame_path,
        frame_id,
        middle_index,
        total_frames,
    )


# ============================================================
# 9. 图像转 Base64
# ============================================================

def image_to_data_uri(
    image_path: str,
) -> str:
    """
    将本地图像转换为 API 可接收的 Data URI。
    """
    extension = (
        os.path.splitext(image_path)[1]
        .lower()
        .lstrip(".")
    )

    if extension == "jpg":
        extension = "jpeg"

    if extension not in {
        "jpeg",
        "png",
        "webp",
    }:
        extension = "jpeg"

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return (
        f"data:image/{extension};"
        f"base64,{encoded}"
    )


# ============================================================
# 10. JSON 解析
# ============================================================

def extract_json_from_text(
    text: str,
) -> Dict:
    """
    尽量从模型输出中提取 JSON。
    """
    text = clean_text(text)

    if not text:
        return {
            "oneline": ""
        }

    # 标准 JSON
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except Exception:
        pass

    # Markdown JSON 代码块
    code_block_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if code_block_match:
        try:
            parsed = json.loads(
                code_block_match.group(1)
            )

            if isinstance(parsed, dict):
                return parsed

        except Exception:
            pass

    # 从其他文字中提取 JSON
    json_match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )

    if json_match:
        try:
            parsed = json.loads(
                json_match.group(0)
            )

            if isinstance(parsed, dict):
                return parsed

        except Exception:
            pass

    # 最后兜底
    return {
        "oneline": text
    }


# ============================================================
# 11. 清理视觉描述
# ============================================================

def sanitize_visual_description(
    text: str,
) -> str:
    """
    清除换行、情绪词和心理推断词。
    """
    text = clean_text(text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    # 删除情绪和心理词
    text = EMOTION_BAN_PATTERN.sub(
        "",
        text,
    )

    # 修复删除后的空格
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    # 修复标点前空格
    text = re.sub(
        r"\s+([,.;:!?])",
        r"\1",
        text,
    )

    # 修复连续逗号或句点
    text = re.sub(
        r",\s*,+",
        ",",
        text,
    )

    text = re.sub(
        r"\.\s*\.+",
        ".",
        text,
    )

    # 删除句首残留标点
    text = text.lstrip(
        " ,.;:"
    ).strip()

    return text


# ============================================================
# 12. API 请求
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
    except requests.RequestException as error:
        raise RuntimeError(
            f"API connection failed: {type(error).__name__}"
        ) from None

    if not response.ok:
        raise RuntimeError(
            f"API request failed with HTTP {response.status_code}"
        )

    try:
        response_json = response.json()
    except ValueError:
        raise RuntimeError(
            "API returned invalid JSON."
        ) from None

    if not isinstance(response_json, dict):
        raise ValueError(
            "API response is not a dictionary."
        )

    return response_json


# ============================================================
# 13. 视觉提示词
# ============================================================

PROMPT = """
You are a visual morphology description assistant for multimodal research.

You are given ONE middle frame from a short spoken video clip.

The video scenes are often visually simple and relatively static. Therefore, pay close attention to fine-grained visible facial morphology and subtle physical cues in this single frame.

Describe only directly observable information, including:

1. Head orientation:
   - facing forward, left, or right
   - slight head tilt
   - chin raised or lowered

2. Gaze:
   - looking toward the camera
   - looking away
   - looking upward, downward, left, or right
   - eyes partly closed, narrowed, or widely open, only when clearly visible

3. Eyebrow morphology:
   - slightly raised
   - lowered
   - drawn together
   - relatively flat or not clearly visible

4. Mouth and lip morphology:
   - mouth open or closed
   - lips lightly or tightly pressed
   - mouth corners slightly raised, lowered, or flat
   - visible asymmetry
   - teeth visible or not visible

5. Facial muscle and cheek cues:
   - cheeks slightly raised
   - visible tension around the mouth or eyes
   - facial shape appearing relaxed only as a physical description, without inferring a mental state

6. Body and gesture cues:
   - upright, leaning forward, leaning backward, or turned sideways
   - visible shoulder position
   - visible hand or arm gestures

STRICT RULES:

- Describe only what can be directly observed in this one frame.
- Do not claim that a cue is brief, temporary, changing, appearing, or disappearing, because only one frame is provided.
- Do not call any cue a micro-expression.
- Do not infer emotion, sentiment, personality, intention, motivation, confidence, psychological state, or cause.
- Do not use emotion or sentiment labels such as positive, negative, neutral, happy, sad, angry, nervous, calm, excited, surprised, or upset.
- Prefer concrete morphology descriptions such as "the mouth corners are slightly raised" instead of expressions such as "the person is smiling happily."
- Do not infer gender, age, ethnicity, identity, occupation, health status, or relationship.
- When a feature is unclear, write "not clearly visible" rather than guessing.
- Produce exactly one concise English sentence.
- Return only one valid JSON object with exactly this structure:

{
  "oneline": "..."
}
""".strip()


# ============================================================
# 14. 调用视觉模型
# ============================================================

def call_gpt_for_image(
    image_path: str,
) -> Optional[Dict[str, str]]:
    """
    为单张中间帧生成一行视觉描述。

    成功：
        {"oneline": "..."}

    所有重试失败：
        None

    失败结果不写入 JSONL，方便之后重新补跑。
    """
    data_uri = image_to_data_uri(
        image_path
    )

    payload = {
        "model": MODEL,
        "temperature": TEMPERATURE,

        # 固定远程生成种子
        "seed": SEED,

        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri,
                            "detail": IMAGE_DETAIL,
                        },
                    },
                ],
            }
        ],
    }

    last_error = None

    for attempt in range(1, RETRY + 1):
        try:
            response_json = post_chat(
                payload
            )

            choices = response_json.get(
                "choices",
                [],
            )

            if not choices:
                raise ValueError(
                    "API response contains no choices."
                )

            message = choices[0].get(
                "message",
                {},
            )

            content = clean_text(
                message.get(
                    "content",
                    "",
                )
            )

            if not content:
                raise ValueError(
                    "The model returned empty content."
                )

            parsed = extract_json_from_text(
                content
            )

            oneline = sanitize_visual_description(
                parsed.get(
                    "oneline",
                    "",
                )
            )

            if not oneline:
                raise ValueError(
                    "The cleaned visual description is empty."
                )

            return {
                "oneline": oneline
            }

        except Exception as error:
            last_error = error

            print(
                f"  [WARN] attempt "
                f"{attempt}/{RETRY} failed:",
                error,
            )

            if attempt < RETRY:
                wait_seconds = (
                    1.0
                    + (attempt - 1) * 0.8
                )

                print(
                    f"  [WAIT] sleep "
                    f"{wait_seconds:.1f} seconds"
                )

                time.sleep(
                    wait_seconds
                )

    print(
        "  [ERROR] all retries failed."
    )
    print(
        "  [ERROR] last error:",
        last_error,
    )

    return None


# ============================================================
# 15. 主流程
# ============================================================

def run() -> None:
    """
    遍历全部 MOSEI clip，选择中间帧并生成视觉描述。
    """
    if not os.path.isdir(ROOT_DIR):
        raise FileNotFoundError(
            f"MOSEI visual root does not exist:\n"
            f"{ROOT_DIR}"
        )

    ensure_output_directory(
        OUT_JSONL
    )

    completed_keys = load_completed_keys(
        OUT_JSONL
    )

    print()
    print("ROOT_DIR:", ROOT_DIR)
    print("OUT_JSONL:", OUT_JSONL)
    print(
        "Already completed:",
        len(completed_keys),
    )

    total_visited = 0
    newly_written = 0
    skipped_completed = 0
    no_frame_count = 0
    failed_count = 0

    with open(
        OUT_JSONL,
        "a",
        encoding="utf-8",
    ) as output_file:

        for row_id, (
            video_id,
            clip_id,
            video_path,
            clip_path,
        ) in enumerate(
            iter_clip_directories(ROOT_DIR)
        ):
            total_visited += 1

            sample_key = make_sample_key(
                video_id,
                clip_id,
            )

            if sample_key in completed_keys:
                skipped_completed += 1
                continue

            (
                frame_path,
                frame_id,
                middle_index,
                total_frames,
            ) = pick_middle_frame(
                clip_path
            )

            if not frame_path:
                no_frame_count += 1

                print()
                print("=" * 72)
                print(
                    f"[WARN] No frame: "
                    f"{video_id}/{clip_id}"
                )

                continue

            print()
            print("=" * 72)
            print(
                f"Item: {row_id + 1}"
            )
            print(
                "video_id:",
                video_id,
            )
            print(
                "clip_id:",
                clip_id,
            )
            print(
                "frame count:",
                total_frames,
            )
            print(
                "middle index:",
                middle_index,
            )
            print(
                "middle frame:",
                frame_path,
            )

            result = call_gpt_for_image(
                frame_path
            )

            # 请求彻底失败时不写入，之后可以继续补跑
            if result is None:
                failed_count += 1

                print(
                    "[WARN] This sample was not written."
                )

                continue

            visual_oneline = result[
                "oneline"
            ]

            output_object = {
                "row_id": row_id,

                # MOSEI 样本标识
                "video_id": video_id,
                "clip_id": clip_id,

                # 目录信息
                "VideoPath": video_path,
                "ClipPath": clip_path,

                # 中间帧信息
                "FrameCount": total_frames,
                "MiddleFrameIndex": middle_index,
                "MiddleFrameId": frame_id,
                "MiddleFramePath": frame_path,

                # 生成配置
                "generation_model": MODEL,
                "generation_seed": SEED,
                "generation_temperature": TEMPERATURE,
                "generation_image_detail": IMAGE_DETAIL,

                # 视觉语义描述
                "visual_oneline": visual_oneline,
            }

            output_file.write(
                json.dumps(
                    output_object,
                    ensure_ascii=False,
                )
                + "\n"
            )

            # 每完成一条立即写入磁盘
            output_file.flush()

            completed_keys.add(
                sample_key
            )

            newly_written += 1

            print(
                "visual_oneline:",
                visual_oneline,
            )

            print(
                "\n[Progress] "
                f"visited={total_visited} | "
                f"completed={len(completed_keys)} | "
                f"new={newly_written} | "
                f"skipped={skipped_completed} | "
                f"no_frame={no_frame_count} | "
                f"failed={failed_count}"
            )

            time.sleep(
                SLEEP_BETWEEN
            )

            if (
                TEST_N is not None
                and TEST_N > 0
                and newly_written >= TEST_N
            ):
                print()
                print(
                    f"Stopped after {TEST_N} "
                    f"new samples."
                )
                print(
                    "Output:",
                    OUT_JSONL,
                )

                return

    print()
    print("=" * 72)
    print("MOSEI visual annotation finished")
    print("=" * 72)

    print(
        "Total visited clips:",
        total_visited,
    )
    print(
        "Newly written:",
        newly_written,
    )
    print(
        "Skipped completed:",
        skipped_completed,
    )
    print(
        "No-frame clips:",
        no_frame_count,
    )
    print(
        "Failed API samples:",
        failed_count,
    )
    print(
        "Output:",
        OUT_JSONL,
    )


# ============================================================
# 16. 启动
# ============================================================

if __name__ == "__main__":
    set_global_seed(SEED)
    run()