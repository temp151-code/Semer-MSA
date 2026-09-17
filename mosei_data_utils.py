#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOSEI six-route data utilities.

Six routes:
    raw text   : text_feature
    LLM text   : text_LLM_feature
    raw audio  : audio_feature
    LLM audio  : audio_LLM_feature
    raw vision : vision_feature_resnet
    LLM vision : visual_LLM_feature

Regression target:
    label_1

Important:
    Regression keeps ALL samples, including Neutral / label_1 == 0.
"""

import pickle
from collections import Counter

import numpy as np
import torch


# ============================================================
# 1. Basic IO
# ============================================================

def load_pkl_items(file_path):
    with open(file_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        return obj

    if isinstance(obj, tuple):
        return list(obj)

    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], (list, tuple)):
            return list(obj["data"])

        raise TypeError(
            "PKL outer object is dict, but no usable 'data' list exists. "
            f"Keys: {list(obj.keys())}"
        )

    raise TypeError(
        f"Unsupported PKL outer type: {type(obj)}"
    )


def _shape_dtype(x):
    if x is None:
        return "MISSING", "MISSING", "None"

    if torch.is_tensor(x):
        return tuple(x.shape), str(x.dtype), type(x).__name__

    if isinstance(x, np.ndarray):
        return tuple(x.shape), str(x.dtype), type(x).__name__

    if hasattr(x, "shape"):
        try:
            return (
                tuple(x.shape),
                str(getattr(x, "dtype", "N/A")),
                type(x).__name__,
            )
        except Exception:
            pass

    return "N/A", "N/A", type(x).__name__


# ============================================================
# 2. Inspection
# ============================================================

def inspect_mosei_items(
    file_path,
    max_print=2,
    start_index=0,
    print_missing=True,
):
    data = load_pkl_items(file_path)

    basic_keys = [
        "video_id",
        "clip_id",
        "label_1",
        "label",
        "mode",
    ]

    feature_keys = [
        "text_feature",
        "text_LLM_feature",
        "audio_feature",
        "audio_LLM_feature",
        "vision_feature_resnet",
        "visual_LLM_feature",
    ]

    end_index = min(
        len(data),
        start_index + max_print,
    )

    for i in range(start_index, end_index):
        item = data[i]

        print("\n" + "=" * 90)
        print(f"Item {i}")

        print("\n[Basic]")
        for key in basic_keys:
            if key in item:
                print(f"  {key}: {item[key]}")
            elif print_missing:
                print(f"  {key}: MISSING")

        print("\n[Features]")
        for key in feature_keys:
            value = item.get(key)
            shape, dtype, type_name = _shape_dtype(value)

            print(
                f"  {key}: "
                f"type={type_name}, "
                f"shape={shape}, "
                f"dtype={dtype}"
            )

    print(
        f"\n[Done] Loaded={len(data)} | "
        f"Printed={end_index - start_index} | "
        f"File={file_path}"
    )

    return data


# ============================================================
# 3. Label encoding
# ============================================================

def encode_regression_labels(
    data,
    label_key="label_1",
):
    labels = []

    for index, item in enumerate(data):
        value = item.get(label_key)

        if value is None:
            raise ValueError(
                f"Item {index} missing regression label: {label_key}"
            )

        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Item {index}: {label_key} cannot be converted to float: "
                f"{value}"
            ) from exc

        if not np.isfinite(value):
            raise ValueError(
                f"Item {index}: {label_key} is not finite: {value}"
            )

        labels.append(value)

    return np.asarray(
        labels,
        dtype=np.float32,
    )


# ============================================================
# 4. Six-route extraction
# ============================================================

def extract_mosei_six_inputs(
    data,
    *,
    task="regression",
    regression_label_key="label_1",
):
    task = str(task).strip().lower()

    if task != "regression":
        raise ValueError(
            "This three-file pipeline is the unified regression protocol. "
            "task must be 'regression'."
        )

    y = encode_regression_labels(
        data,
        label_key=regression_label_key,
    )

    X_text_raw = []
    X_text_llm = []

    X_audio_raw = []
    X_audio_llm = []

    X_vision_raw = []
    X_vision_llm = []

    sample_keys = []

    missing = Counter()
    used_keys = Counter()
    status_counts = Counter()

    for item in data:
        video_id = str(item.get("video_id", "")).strip()
        clip_id = str(item.get("clip_id", "")).strip()

        sample_keys.append(
            (video_id, clip_id)
        )

        field_map = [
            ("text_raw", "text_feature", X_text_raw),
            ("text_llm", "text_LLM_feature", X_text_llm),
            ("audio_raw", "audio_feature", X_audio_raw),
            ("audio_llm", "audio_LLM_feature", X_audio_llm),
            ("vision_raw", "vision_feature_resnet", X_vision_raw),
            ("vision_llm", "visual_LLM_feature", X_vision_llm),
        ]

        for route_name, field_name, output_list in field_map:
            value = item.get(field_name)

            if value is None:
                missing[route_name] += 1
            else:
                used_keys[f"{route_name}::{field_name}"] += 1

            output_list.append(value)

        for status_key in (
            "text_LLM_status",
            "audio_LLM_status",
            "visual_LLM_status",
        ):
            status = str(
                item.get(status_key, "MISSING")
            ).strip()

            status_counts[
                f"{status_key}::{status}"
            ] += 1

    return {
        "y": y,
        "sample_keys": sample_keys,

        "X_text_raw": X_text_raw,
        "X_text_llm": X_text_llm,

        "X_audio_raw": X_audio_raw,
        "X_audio_llm": X_audio_llm,

        "X_vision_raw": X_vision_raw,
        "X_vision_llm": X_vision_llm,

        "missing": dict(missing),
        "used_keys": dict(used_keys),
        "status_counts": dict(status_counts),

        "task": "regression",
    }


# ============================================================
# 5. Checks
# ============================================================

def check_duplicate_sample_keys(data):
    keys = [
        (
            str(item.get("video_id", "")).strip(),
            str(item.get("clip_id", "")).strip(),
        )
        for item in data
    ]

    counts = Counter(keys)

    duplicates = {
        key: count
        for key, count in counts.items()
        if count > 1
    }

    print(
        f"Samples={len(keys)} | "
        f"Unique keys={len(counts)} | "
        f"Duplicate keys={len(duplicates)}"
    )

    if duplicates:
        print("First duplicate keys:")
        for key, count in list(duplicates.items())[:20]:
            print(f"  {key}: {count}")

    return duplicates


def check_six_input_completeness(data):
    feature_fields = {
        "text_raw": "text_feature",
        "text_llm": "text_LLM_feature",
        "audio_raw": "audio_feature",
        "audio_llm": "audio_LLM_feature",
        "vision_raw": "vision_feature_resnet",
        "vision_llm": "visual_LLM_feature",
    }

    print("\nSix-route completeness:")

    out = {}

    for route, field in feature_fields.items():
        missing = sum(
            item.get(field) is None
            for item in data
        )

        out[route] = missing

        print(
            f"  {route:12s}: "
            f"missing={missing}/{len(data)} "
            f"({missing / max(len(data), 1):.4%})"
        )

    return out


def check_mode_distribution(data):
    counts = Counter(
        str(item.get("mode", "MISSING")).strip()
        for item in data
    )

    print(
        "Mode distribution:",
        dict(counts),
    )

    return counts


def print_label_sign_distribution(
    name,
    y,
):
    y = np.asarray(y)

    neg = int((y < 0).sum())
    zero = int((y == 0).sum())
    pos = int((y > 0).sum())

    print(
        f"{name} sign distribution: "
        f"negative={neg}, zero={zero}, positive={pos}"
    )


def print_pack_summary(
    name,
    pack,
):
    y = np.asarray(pack["y"])

    print("\n" + "=" * 80)
    print(f"{name} regression pack")
    print("=" * 80)

    print("Task:", pack["task"])
    print("Samples:", len(y))
    print("y shape:", y.shape)
    print("y dtype:", y.dtype)

    print("Label min:", float(np.min(y)))
    print("Label max:", float(np.max(y)))
    print("Label mean:", float(np.mean(y)))

    print_label_sign_distribution(
        name,
        y,
    )

    print("Missing:", pack["missing"])
    print("Used keys:", pack["used_keys"])
    print("Status counts:", pack["status_counts"])

    expected = len(y)

    for key in (
        "sample_keys",
        "X_text_raw",
        "X_text_llm",
        "X_audio_raw",
        "X_audio_llm",
        "X_vision_raw",
        "X_vision_llm",
    ):
        actual = len(pack[key])

        if actual != expected:
            raise RuntimeError(
                f"{name}: {key} length mismatch: "
                f"{actual} != {expected}"
            )

    print("✅ All six-route lengths are aligned.")


# ============================================================
# 6. Feature tensor normalization
# ============================================================

def prepare_feature_list(
    feature_list,
    *,
    modality,
    dtype=torch.float32,
):
    """
    Convert route features to:
        List[Tensor(L,D) | None]

    Rules:
        (1,L,D) -> (L,D)
        (D,)     -> (1,D)
    """

    out = []

    for feat in feature_list:
        if feat is None:
            out.append(None)
            continue

        if torch.is_tensor(feat):
            x = feat.detach().cpu().to(dtype)
        else:
            x = torch.tensor(
                feat,
                dtype=dtype,
            )

        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        if x.dim() != 2:
            raise ValueError(
                f"Unexpected feature shape for {modality}: "
                f"{tuple(x.shape)}"
            )

        out.append(x.contiguous())

    return out


def preprocess_pack(pack):
    """
    Rename/normalize routes to the fields consumed by the training script.
    """

    new_pack = dict(pack)

    new_pack["X_text_raw"] = prepare_feature_list(
        pack["X_text_raw"],
        modality="text_raw",
    )

    new_pack["X_text_sem"] = prepare_feature_list(
        pack["X_text_llm"],
        modality="text_llm",
    )

    new_pack["X_audio"] = prepare_feature_list(
        pack["X_audio_raw"],
        modality="audio_raw",
    )

    new_pack["X_audio_llm"] = prepare_feature_list(
        pack["X_audio_llm"],
        modality="audio_llm",
    )

    new_pack["X_vis_raw"] = prepare_feature_list(
        pack["X_vision_raw"],
        modality="vision_raw",
    )

    new_pack["X_vis_cap"] = prepare_feature_list(
        pack["X_vision_llm"],
        modality="vision_llm",
    )

    return new_pack
