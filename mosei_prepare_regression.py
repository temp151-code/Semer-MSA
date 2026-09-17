#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare MOSEI train / valid / test regression packs.

Important:
    - ALL samples are kept.
    - Neutral / label_1 == 0 is NOT removed.
    - The same continuous regression target label_1 is used for training.
    - Acc-2 / F1 / Acc-7 are derived later from the regression predictions.
"""

import argparse

from mosei_data_utils import (
    inspect_mosei_items,
    load_pkl_items,
    check_duplicate_sample_keys,
    check_six_input_completeness,
    check_mode_distribution,
    extract_mosei_six_inputs,
    print_pack_summary,
    preprocess_pack,
)


DEFAULT_BASE_DIR = (
    "/root/autodl-tmp/"
    "LLMbaseddataaugmentation"
)

DEFAULT_TRAIN_PATH = (
    f"{DEFAULT_BASE_DIR}/"
    "MOSEI_train_audio_text_visual_LLM_seed42.pkl"
)

DEFAULT_VALID_PATH = (
    f"{DEFAULT_BASE_DIR}/"
    "MOSEI_valid_LLM_newaudio_train_threshold_seed42.pkl"
)

DEFAULT_TEST_PATH = (
    f"{DEFAULT_BASE_DIR}/"
    "MOSEI_test_LLM_newaudio_train_threshold_seed42.pkl"
)


def _check_expected_mode(
    name,
    data,
    expected_mode,
):
    seen = {
        str(item.get("mode", "")).strip()
        for item in data
    }

    seen.discard("")

    if seen and seen != {expected_mode}:
        print(
            f"⚠️ {name}: expected mode='{expected_mode}', "
            f"but found {sorted(seen)}"
        )


def build_regression_packs(
    train_path=DEFAULT_TRAIN_PATH,
    valid_path=DEFAULT_VALID_PATH,
    test_path=DEFAULT_TEST_PATH,
    *,
    inspect_samples=0,
    run_checks=True,
):
    """
    Return:
        train_data, valid_data, test_data,
        train_pack2, valid_pack2, test_pack2
    """

    print("=" * 80)
    print("MOSEI unified regression preparation")
    print("=" * 80)

    if inspect_samples > 0:
        print("\n[Inspect TRAIN]")
        train_data = inspect_mosei_items(
            train_path,
            max_print=inspect_samples,
        )

        print("\n[Inspect VALID]")
        valid_data = inspect_mosei_items(
            valid_path,
            max_print=inspect_samples,
        )

        print("\n[Inspect TEST]")
        test_data = inspect_mosei_items(
            test_path,
            max_print=inspect_samples,
        )

    else:
        train_data = load_pkl_items(train_path)
        valid_data = load_pkl_items(valid_path)
        test_data = load_pkl_items(test_path)

    print(
        "\nRaw split sizes:",
        f"train={len(train_data)} | "
        f"valid={len(valid_data)} | "
        f"test={len(test_data)}"
    )

    _check_expected_mode(
        "TRAIN",
        train_data,
        "train",
    )

    _check_expected_mode(
        "VALID",
        valid_data,
        "valid",
    )

    _check_expected_mode(
        "TEST",
        test_data,
        "test",
    )

    if run_checks:
        for name, data in (
            ("TRAIN", train_data),
            ("VALID", valid_data),
            ("TEST", test_data),
        ):
            print("\n" + "-" * 80)
            print(name)
            print("-" * 80)

            check_duplicate_sample_keys(data)
            check_mode_distribution(data)
            check_six_input_completeness(data)

    # ========================================================
    # Regression packs
    #
    # DO NOT remove Neutral.
    # ========================================================

    train_pack = extract_mosei_six_inputs(
        train_data,
        task="regression",
        regression_label_key="label_1",
    )

    valid_pack = extract_mosei_six_inputs(
        valid_data,
        task="regression",
        regression_label_key="label_1",
    )

    test_pack = extract_mosei_six_inputs(
        test_data,
        task="regression",
        regression_label_key="label_1",
    )

    print_pack_summary(
        "TRAIN",
        train_pack,
    )

    print_pack_summary(
        "VALID",
        valid_pack,
    )

    print_pack_summary(
        "TEST",
        test_pack,
    )

    print("\nNormalize six feature routes...")

    train_pack2 = preprocess_pack(train_pack)
    valid_pack2 = preprocess_pack(valid_pack)
    test_pack2 = preprocess_pack(test_pack)

    print("✅ Regression packs ready.")
    print("✅ Neutral samples are retained.")

    return (
        train_data,
        valid_data,
        test_data,
        train_pack2,
        valid_pack2,
        test_pack2,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare MOSEI six-route unified regression packs."
    )

    parser.add_argument(
        "--train-path",
        type=str,
        default=DEFAULT_TRAIN_PATH,
    )

    parser.add_argument(
        "--valid-path",
        type=str,
        default=DEFAULT_VALID_PATH,
    )

    parser.add_argument(
        "--test-path",
        type=str,
        default=DEFAULT_TEST_PATH,
    )

    parser.add_argument(
        "--inspect-samples",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--skip-checks",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    build_regression_packs(
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        inspect_samples=args.inspect_samples,
        run_checks=not args.skip_checks,
    )


if __name__ == "__main__":
    main()
