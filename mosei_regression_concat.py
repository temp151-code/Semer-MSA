#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MOSEI six-route CONCAT unified regression training.

============================================================
Data protocol
============================================================

Keep ALL MOSEI samples:
    Negative
    Neutral (label_1 == 0)
    Positive

Train ONE continuous regressor on:
    label_1

Six routes:
    raw text   + LLM text
    raw audio  + LLM audio
    raw vision + LLM vision

Within each modality:
    raw tokens + LLM tokens
        -> CONCAT
        -> attention pooling
        -> context Transformer

Then:
    text / vision / audio regression heads
        -> 3-way regression voter
        -> one continuous prediction

============================================================
Reported metrics
============================================================

Continuous:
    MAE                 lower is better
    Pearson Corr        higher is better

7-class:
    Acc-7

Binary derived from SAME regression prediction:
    Has0 Acc-2 / weighted F1
        negative vs non-negative
        includes truth == 0

    Non0 Acc-2 / weighted F1
        negative vs positive
        removes truth == 0

Attention please! This is NOT a separately trained binary classifier.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from torch.utils.data import (
    Dataset,
    DataLoader,
)
from torch.cuda.amp import GradScaler

from mosei_prepare_regression import (
    build_regression_packs,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VALID_PATH,
    DEFAULT_TEST_PATH,
)


# ============================================================
# 0. Defaults
# ============================================================

DEFAULT_SEED = 42
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 8
DEFAULT_CONTEXT_SIZE = 5

DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_GRAD_CLIP = 1.0

DEFAULT_D_MODEL = 768
DEFAULT_DROPOUT = 0.1

DEFAULT_OUTPUT_DIR = (
    "/root/autodl-tmp/"
    "LLMbaseddataaugmentation/"
    "mosei_concat_regression_outputs"
)


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Context dataset
# ============================================================

class ContextWindowDataset(Dataset):
    """
    Context window:
        C=5 -> [t-2, t-1, t, t+1, t+2]

    Context never crosses video_id boundaries.
    """

    def __init__(
        self,
        pack,
        meta_data,
        C=5,
        use_or_for_text_exist=True,
    ):
        super().__init__()

        if C % 2 != 1:
            raise ValueError(
                "Context size C must be odd."
            )

        self.C = int(C)
        self.r = self.C // 2
        self.use_or_for_text_exist = bool(
            use_or_for_text_exist
        )

        self.y = pack["y"]

        self.text_raw = pack["X_text_raw"]
        self.text_sem = pack["X_text_sem"]

        self.audio_raw = pack["X_audio"]
        self.audio_llm = pack["X_audio_llm"]

        self.vis_raw = pack["X_vis_raw"]
        self.vis_cap = pack["X_vis_cap"]

        self.meta = meta_data

        n = len(self.y)

        if len(self.meta) != n:
            raise ValueError(
                f"meta_data and labels mismatch: "
                f"{len(self.meta)} != {n}"
            )

        feature_lists = {
            "X_text_raw": self.text_raw,
            "X_text_sem": self.text_sem,
            "X_audio": self.audio_raw,
            "X_audio_llm": self.audio_llm,
            "X_vis_raw": self.vis_raw,
            "X_vis_cap": self.vis_cap,
        }

        for name, values in feature_lists.items():
            if len(values) != n:
                raise ValueError(
                    f"{name} length mismatch: "
                    f"{len(values)} != {n}"
                )

        videos = defaultdict(list)

        for idx, item in enumerate(self.meta):
            video_id = str(
                item.get("video_id", "")
            ).strip()

            clip_id = item.get(
                "clip_id",
                None,
            )

            if not video_id:
                raise ValueError(
                    f"Sample {idx} missing video_id."
                )

            if clip_id is None:
                raise ValueError(
                    f"Sample {idx} missing clip_id."
                )

            videos[video_id].append(
                (clip_id, idx)
            )

        def clip_sort_key(pair):
            clip_id, _ = pair

            try:
                return (0, float(clip_id))
            except (TypeError, ValueError):
                return (1, str(clip_id))

        self.videos = {}
        self.pos_in_video = {}

        for video_id, items in videos.items():
            items_sorted = sorted(
                items,
                key=clip_sort_key,
            )

            indices = [
                idx
                for _, idx in items_sorted
            ]

            self.videos[video_id] = indices

            for position, idx in enumerate(indices):
                self.pos_in_video[idx] = position

        self.video_id_of = [
            str(
                item.get("video_id", "")
            ).strip()
            for item in self.meta
        ]

    def __len__(self):
        return len(self.y)

    def _get_one(self, idx):
        tr = self.text_raw[idx]
        ts = self.text_sem[idx]

        ar = self.audio_raw[idx]
        al = self.audio_llm[idx]

        vr = self.vis_raw[idx]
        vc = self.vis_cap[idx]

        if self.use_or_for_text_exist:
            m_t = 0.0 if (
                tr is None
                and ts is None
            ) else 1.0
        else:
            m_t = 0.0 if tr is None else 1.0

        m_ar = 0.0 if ar is None else 1.0
        m_al = 0.0 if al is None else 1.0
        m_v = 0.0 if vr is None else 1.0

        modal_mask = torch.tensor(
            [
                m_t,
                m_ar,
                m_al,
                m_v,
            ],
            dtype=torch.float32,
        )

        return (
            tr,
            ts,
            ar,
            al,
            vr,
            vc,
            modal_mask,
        )

    def __getitem__(self, center_idx):
        video_id = self.video_id_of[
            center_idx
        ]

        video_indices = self.videos[
            video_id
        ]

        center_position = self.pos_in_video[
            center_idx
        ]

        window_indices = []
        context_mask = []

        for offset in range(
            -self.r,
            self.r + 1,
        ):
            position = (
                center_position
                + offset
            )

            if (
                0
                <= position
                < len(video_indices)
            ):
                window_indices.append(
                    video_indices[position]
                )
                context_mask.append(True)
            else:
                window_indices.append(None)
                context_mask.append(False)

        text_raw_list = []
        text_sem_list = []

        audio_raw_list = []
        audio_llm_list = []

        vision_raw_list = []
        vision_cap_list = []

        modal_mask_list = []

        for window_idx in window_indices:
            if window_idx is None:
                text_raw_list.append(None)
                text_sem_list.append(None)

                audio_raw_list.append(None)
                audio_llm_list.append(None)

                vision_raw_list.append(None)
                vision_cap_list.append(None)

                modal_mask_list.append(
                    torch.zeros(
                        4,
                        dtype=torch.float32,
                    )
                )

            else:
                (
                    tr,
                    ts,
                    ar,
                    al,
                    vr,
                    vc,
                    modal_mask,
                ) = self._get_one(
                    window_idx
                )

                text_raw_list.append(tr)
                text_sem_list.append(ts)

                audio_raw_list.append(ar)
                audio_llm_list.append(al)

                vision_raw_list.append(vr)
                vision_cap_list.append(vc)

                modal_mask_list.append(
                    modal_mask
                )

        y = torch.tensor(
            float(
                self.y[center_idx]
            ),
            dtype=torch.float32,
        )

        return {
            "text_raw_list":
                text_raw_list,

            "text_sem_list":
                text_sem_list,

            "audio_raw_list":
                audio_raw_list,

            "audio_llm_list":
                audio_llm_list,

            "vision_raw_list":
                vision_raw_list,

            "vision_cap_list":
                vision_cap_list,

            "ctx_mask":
                torch.tensor(
                    context_mask,
                    dtype=torch.bool,
                ),

            "center_pos":
                torch.tensor(
                    self.r,
                    dtype=torch.long,
                ),

            "modal_mask":
                torch.stack(
                    modal_mask_list,
                    dim=0,
                ),

            "y":
                y,
        }


# ============================================================
# 3. Fast collate
# ============================================================

def collate_context_multi(
    batch,
    C=5,
):
    """
    Single-pass global padding.

    Output:
        text_x       [B,C,Lt,768]
        sem_x        [B,C,Ls,768]
        audio_x      [B,C,La,1024]
        audio_llm_x  [B,C,Lal,768]
        vision_x     [B,C,Lv,2048]
        cap_x        [B,C,Lc,768]
    """

    B = len(batch)

    out = {
        "y": torch.stack(
            [b["y"] for b in batch],
            dim=0,
        ),

        "ctx_mask": torch.stack(
            [b["ctx_mask"] for b in batch],
            dim=0,
        ),

        "center_idx": torch.stack(
            [b["center_pos"] for b in batch],
            dim=0,
        ),

        "modal_mask": torch.stack(
            [b["modal_mask"] for b in batch],
            dim=0,
        ),
    }

    def pad_context_fast(
        list_name,
        feat_dim,
        out_x_key,
        out_mask_key,
        dtype=torch.float32,
    ):
        seqs = []

        for b in batch:
            seq_list = b[list_name]

            if len(seq_list) != C:
                raise ValueError(
                    f"{list_name}: "
                    f"expected C={C}, "
                    f"got {len(seq_list)}"
                )

            seqs.extend(
                seq_list
            )

        lengths = [
            0
            if x is None
            else int(x.size(0))
            for x in seqs
        ]

        L_global = max(
            max(
                lengths,
                default=0,
            ),
            1,
        )

        padded = torch.zeros(
            B * C,
            L_global,
            feat_dim,
            dtype=dtype,
        )

        valid_mask = torch.zeros(
            B * C,
            L_global,
            dtype=torch.bool,
        )

        for i, (x, L) in enumerate(
            zip(seqs, lengths)
        ):
            if x is None or L == 0:
                continue

            if x.dim() != 2:
                raise ValueError(
                    f"{list_name}: "
                    f"expected 2D tensor, "
                    f"got {tuple(x.shape)}"
                )

            if x.size(1) != feat_dim:
                raise ValueError(
                    f"{list_name}: "
                    f"expected D={feat_dim}, "
                    f"got {tuple(x.shape)}"
                )

            if x.dtype != dtype:
                x = x.to(dtype)

            padded[
                i,
                :L,
                :
            ].copy_(x)

            valid_mask[
                i,
                :L
            ] = True

        out[out_x_key] = padded.view(
            B,
            C,
            L_global,
            feat_dim,
        )

        out[out_mask_key] = valid_mask.view(
            B,
            C,
            L_global,
        )

    pad_context_fast(
        "text_raw_list",
        768,
        "text_x",
        "text_mask",
    )

    pad_context_fast(
        "text_sem_list",
        768,
        "sem_x",
        "sem_mask",
    )

    pad_context_fast(
        "audio_raw_list",
        1024,
        "audio_x",
        "audio_mask",
    )

    pad_context_fast(
        "audio_llm_list",
        768,
        "audio_llm_x",
        "audio_llm_mask",
    )

    pad_context_fast(
        "vision_raw_list",
        2048,
        "vision_x",
        "vision_mask",
    )

    pad_context_fast(
        "vision_cap_list",
        768,
        "cap_x",
        "cap_mask",
    )

    return out


def build_loaders(
    train_pack,
    valid_pack,
    test_pack,
    train_data,
    valid_data,
    test_data,
    *,
    C=5,
    batch_size=8,
    num_workers=0,
):
    train_dataset = ContextWindowDataset(
        train_pack,
        train_data,
        C=C,
    )

    valid_dataset = ContextWindowDataset(
        valid_pack,
        valid_data,
        C=C,
    )

    test_dataset = ContextWindowDataset(
        test_pack,
        test_data,
        C=C,
    )

    def collate_fn(batch):
        return collate_context_multi(
            batch,
            C=C,
        )

    common = {
        "batch_size":
            batch_size,

        "num_workers":
            num_workers,

        "collate_fn":
            collate_fn,

        "pin_memory":
            True,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common,
    )

    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        **common,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common,
    )

    print(
        "\nDataLoader:",
        f"train={len(train_dataset)} | "
        f"valid={len(valid_dataset)} | "
        f"test={len(test_dataset)} | "
        f"batch={batch_size} | "
        f"C={C} | "
        f"workers={num_workers} | "
        f"pin_memory=True"
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
    )


# ============================================================
# 4. Model components
# ============================================================

class SafeAttnPool(nn.Module):
    """
    mask:
        True = valid
        False = padding
    """

    def __init__(
        self,
        d_model=768,
        dropout=0.1,
    ):
        super().__init__()

        self.scorer = nn.Linear(
            d_model,
            1,
        )

        self.drop = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        mask,
    ):
        N, L, D = x.shape
        x_dtype = x.dtype

        with torch.autocast(
            device_type=x.device.type,
            enabled=False,
        ):
            x32 = x.float()

            s = self.scorer(
                self.drop(
                    x32
                )
            ).squeeze(-1)

            valid_row = mask.any(
                dim=1
            )

            out32 = x32.new_zeros(
                N,
                D,
            )

            if valid_row.any():
                xv = x32[
                    valid_row
                ]

                mv = mask[
                    valid_row
                ]

                sv = s[
                    valid_row
                ].masked_fill(
                    ~mv,
                    float("-inf"),
                )

                a = torch.softmax(
                    sv,
                    dim=-1,
                )

                out_v = torch.bmm(
                    a.unsqueeze(1),
                    xv,
                ).squeeze(1)

                out32[
                    valid_row
                ] = out_v

        return out32.to(
            dtype=x_dtype
        )


class AudioDownsampler(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        out_dim=768,
        kernel=5,
        stride=2,
        dropout=0.1,
    ):
        super().__init__()

        if kernel % 2 != 1:
            raise ValueError(
                "Audio kernel should be odd."
            )

        self.kernel = int(kernel)
        self.stride = int(stride)
        self.pad = self.kernel // 2

        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.pad,
        )

        self.act = nn.GELU()
        self.drop = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        mask,
    ):
        y = self.conv(
            x.transpose(1, 2)
        )

        y = self.drop(
            self.act(y)
        ).transpose(1, 2)

        m_ds = F.max_pool1d(
            mask.float().unsqueeze(1),
            kernel_size=self.kernel,
            stride=self.stride,
            padding=self.pad,
        ).squeeze(1)

        m_ds = (
            m_ds > 0.5
        )

        if y.size(1) != m_ds.size(1):
            L = min(
                y.size(1),
                m_ds.size(1),
            )

            y = y[:, :L]
            m_ds = m_ds[:, :L]

        return y, m_ds


# ============================================================
# 5. CONCAT unified regressor
# ============================================================

class ContextEmotionTriModalLLMRegressor(
    nn.Module
):
    """
    Utterance level:
        Text:
            concat(raw text, LLM text)
            -> attention pool

        Vision:
            raw vision 2048 -> 768
            concat(raw vision, LLM vision)
            -> attention pool

        Audio:
            raw audio 1024 -> 768 with learnable Conv1d downsampling
            concat(raw audio, LLM audio)
            -> attention pool

    Context level:
        shared TransformerEncoder is applied to
        text / vision / audio context sequences separately.

    Output:
        three modality regression scores
        + learned 3-way voter
        -> one continuous sentiment prediction.
    """

    def __init__(
        self,
        d_model=768,
        dropout=0.1,
        ctx_layers=1,
        ctx_heads=8,
        audio_in_dim=1024,
        ds_kernel=5,
        ds_stride=2,
        voter_temperature=1.0,
    ):
        super().__init__()

        self.voter_temperature = float(
            voter_temperature
        )

        self.pool_text = SafeAttnPool(
            d_model,
            dropout=dropout,
        )

        self.proj_vis = nn.Linear(
            2048,
            d_model,
        )

        self.pool_vis = SafeAttnPool(
            d_model,
            dropout=dropout,
        )

        self.audio_ds = AudioDownsampler(
            in_dim=audio_in_dim,
            out_dim=d_model,
            kernel=ds_kernel,
            stride=ds_stride,
            dropout=dropout,
        )

        self.pool_aud = SafeAttnPool(
            d_model,
            dropout=dropout,
        )

        self.miss_text = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        self.miss_sem = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        self.miss_vis = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        self.miss_cap = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        self.miss_aud = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        self.miss_audllm = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=ctx_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )

        self.ctx_encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=ctx_layers,
        )

        self.reg_text = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        self.reg_vis = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        self.reg_aud = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        self.regression_query = nn.Parameter(
            torch.randn(
                1,
                d_model,
            )
            * 0.02
        )

        self.WQ = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.WK = nn.Linear(
            d_model,
            d_model,
            bias=False,
        )

        self.last_W = None

    @staticmethod
    def _apply_missing_token(
        x,
        mask,
        valid_u,
        miss_token,
    ):
        need = (
            (~valid_u)
            | (~mask.any(dim=1))
        )

        if need.any():
            x = x.clone()
            mask = mask.clone()

            x[need] = 0.0

            x[
                need,
                0:1,
                :
            ] = miss_token.to(
                dtype=x.dtype,
                device=x.device,
            )

            mask[need] = False
            mask[need, 0] = True

        return x, mask

    def _regression_voter_weights(
        self,
        ct,
        cv,
        ca,
    ):
        B, D = ct.shape

        q = self.WQ(
            self.regression_query
        )

        q = q.expand(
            B,
            -1,
        ).unsqueeze(1)

        Fm = torch.stack(
            [
                ct,
                cv,
                ca,
            ],
            dim=1,
        )

        Kmat = self.WK(Fm)

        att = torch.matmul(
            q,
            Kmat.transpose(1, 2),
        ).squeeze(1)

        att = (
            att
            / math.sqrt(D)
        )

        if self.voter_temperature != 1.0:
            att = (
                att
                / self.voter_temperature
            )

        return torch.softmax(
            att,
            dim=-1,
        )

    def forward(
        self,
        batch,
        return_debug=False,
    ):
        text_x = batch["text_x"]
        text_mask = batch["text_mask"]

        sem_x = batch["sem_x"]
        sem_mask = batch["sem_mask"]

        vision_x = batch["vision_x"]
        vision_mask = batch["vision_mask"]

        cap_x = batch["cap_x"]
        cap_mask = batch["cap_mask"]

        audio_x = batch["audio_x"]
        audio_mask = batch["audio_mask"]

        audio_llm_x = batch[
            "audio_llm_x"
        ]

        audio_llm_mask = batch[
            "audio_llm_mask"
        ]

        ctx_mask = batch["ctx_mask"]
        center_idx = batch["center_idx"]

        B, C, Lt, D = text_x.shape

        N = B * C

        # Text
        t = text_x.reshape(
            N,
            Lt,
            D,
        )

        tm = text_mask.reshape(
            N,
            Lt,
        )

        Ls = sem_x.size(2)

        s = sem_x.reshape(
            N,
            Ls,
            D,
        )

        sm = sem_mask.reshape(
            N,
            Ls,
        )

        # Vision
        Lv = vision_x.size(2)

        v = vision_x.reshape(
            N,
            Lv,
            2048,
        )

        vm = vision_mask.reshape(
            N,
            Lv,
        )

        Lc = cap_x.size(2)

        c = cap_x.reshape(
            N,
            Lc,
            D,
        )

        cm = cap_mask.reshape(
            N,
            Lc,
        )

        # Audio
        La = audio_x.size(2)

        a = audio_x.reshape(
            N,
            La,
            1024,
        )

        am = audio_mask.reshape(
            N,
            La,
        )

        Lal = audio_llm_x.size(2)

        al = audio_llm_x.reshape(
            N,
            Lal,
            D,
        )

        alm = audio_llm_mask.reshape(
            N,
            Lal,
        )

        valid_u = ctx_mask.reshape(
            N
        )

        # Missing stabilization
        t, tm = self._apply_missing_token(
            t,
            tm,
            valid_u,
            self.miss_text,
        )

        s, sm = self._apply_missing_token(
            s,
            sm,
            valid_u,
            self.miss_sem,
        )

        v = self.proj_vis(v)

        v, vm = self._apply_missing_token(
            v,
            vm,
            valid_u,
            self.miss_vis,
        )

        c, cm = self._apply_missing_token(
            c,
            cm,
            valid_u,
            self.miss_cap,
        )

        al, alm = self._apply_missing_token(
            al,
            alm,
            valid_u,
            self.miss_audllm,
        )

        a, am = self.audio_ds(
            a,
            am,
        )

        a, am = self._apply_missing_token(
            a,
            am,
            valid_u,
            self.miss_aud,
        )

        # ====================================================
        # CONCAT
        # ====================================================

        t_cat = torch.cat(
            [t, s],
            dim=1,
        )

        tm_cat = torch.cat(
            [tm, sm],
            dim=1,
        )

        u_text = self.pool_text(
            t_cat,
            tm_cat,
        ).reshape(
            B,
            C,
            D,
        )

        v_cat = torch.cat(
            [v, c],
            dim=1,
        )

        vm_cat = torch.cat(
            [vm, cm],
            dim=1,
        )

        u_vis = self.pool_vis(
            v_cat,
            vm_cat,
        ).reshape(
            B,
            C,
            D,
        )

        a_cat = torch.cat(
            [a, al],
            dim=1,
        )

        am_cat = torch.cat(
            [am, alm],
            dim=1,
        )

        u_aud = self.pool_aud(
            a_cat,
            am_cat,
        ).reshape(
            B,
            C,
            D,
        )

        # Context
        u_text_ctx = self.ctx_encoder(
            u_text,
            src_key_padding_mask=(
                ~ctx_mask
            ),
        )

        u_vis_ctx = self.ctx_encoder(
            u_vis,
            src_key_padding_mask=(
                ~ctx_mask
            ),
        )

        u_aud_ctx = self.ctx_encoder(
            u_aud,
            src_key_padding_mask=(
                ~ctx_mask
            ),
        )

        idx = center_idx.clamp(
            0,
            C - 1,
        ).view(
            B,
            1,
            1,
        ).expand(
            B,
            1,
            D,
        )

        ct = torch.gather(
            u_text_ctx,
            dim=1,
            index=idx,
        ).squeeze(1)

        cv = torch.gather(
            u_vis_ctx,
            dim=1,
            index=idx,
        ).squeeze(1)

        ca = torch.gather(
            u_aud_ctx,
            dim=1,
            index=idx,
        ).squeeze(1)

        s_t = self.reg_text(
            ct
        ).squeeze(-1)

        s_v = self.reg_vis(
            cv
        ).squeeze(-1)

        s_a = self.reg_aud(
            ca
        ).squeeze(-1)

        S = torch.stack(
            [s_t, s_v, s_a],
            dim=-1,
        )

        W = self._regression_voter_weights(
            ct,
            cv,
            ca,
        )

        self.last_W = W.detach()

        pred = (
            W * S
        ).sum(dim=-1)

        if not return_debug:
            return pred

        return (
            pred,
            {
                "W": W.detach(),
                "S": S.detach(),
                "center_repr": {
                    "text": ct.detach(),
                    "vis": cv.detach(),
                    "aud": ca.detach(),
                },
            },
        )


# ============================================================
# 6. Metrics
# ============================================================

def regression_sentiment_metrics(
    y_true,
    y_pred,
):
    """
    Has0:
        keep truth == 0
        negative vs non-negative

    Non0:
        remove truth == 0
        negative vs positive

    Acc-7:
        clip prediction / truth to [-3,3], then round.
    """

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if len(y_true) == 0:
        raise ValueError(
            "Empty y_true."
        )

    mae = float(
        np.mean(
            np.abs(
                y_true - y_pred
            )
        )
    )

    if (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        corr = float(
            np.corrcoef(
                y_pred,
                y_true,
            )[0, 1]
        )
    else:
        corr = float("nan")

    # -----------------------
    # Acc-7
    # -----------------------

    y_true_7 = np.round(
        np.clip(
            y_true,
            -3.0,
            3.0,
        )
    )

    y_pred_7 = np.round(
        np.clip(
            y_pred,
            -3.0,
            3.0,
        )
    )

    acc7 = float(
        accuracy_score(
            y_true_7,
            y_pred_7,
        )
    )

    # -----------------------
    # Has0 Acc-2 / F1
    # negative vs non-negative
    # -----------------------

    has0_true = (
        y_true >= 0
    )

    has0_pred = (
        y_pred >= 0
    )

    has0_acc2 = float(
        accuracy_score(
            has0_true,
            has0_pred,
        )
    )

    has0_f1 = float(
        f1_score(
            has0_true,
            has0_pred,
            average="weighted",
            zero_division=0,
        )
    )

    # -----------------------
    # Non0 Acc-2 / F1
    # -----------------------

    non0_mask = (
        y_true != 0
    )

    if non0_mask.any():
        non0_true = (
            y_true[
                non0_mask
            ] > 0
        )

        non0_pred = (
            y_pred[
                non0_mask
            ] > 0
        )

        non0_acc2 = float(
            accuracy_score(
                non0_true,
                non0_pred,
            )
        )

        non0_f1 = float(
            f1_score(
                non0_true,
                non0_pred,
                average="weighted",
                zero_division=0,
            )
        )

        non0_n = int(
            non0_mask.sum()
        )

    else:
        non0_acc2 = float("nan")
        non0_f1 = float("nan")
        non0_n = 0

    return {
        "MAE": mae,
        "Corr": corr,
        "Acc7": acc7,

        "Has0_Acc2":
            has0_acc2,

        "Has0_F1":
            has0_f1,

        "Non0_Acc2":
            non0_acc2,

        "Non0_F1":
            non0_f1,

        "N":
            int(len(y_true)),

        "Non0_N":
            non0_n,
    }


# ============================================================
# 7. Train / eval
# ============================================================

def move_batch_to_device(
    batch,
    device,
):
    if torch.is_tensor(batch):
        return batch.to(
            device,
            non_blocking=True,
        )

    if isinstance(batch, dict):
        return {
            k:
                move_batch_to_device(
                    v,
                    device,
                )
            for k, v in batch.items()
        }

    if isinstance(
        batch,
        (list, tuple),
    ):
        return type(batch)(
            move_batch_to_device(
                x,
                device,
            )
            for x in batch
        )

    return batch


def run_one_epoch(
    model,
    loader,
    criterion,
    device,
    *,
    optimizer=None,
    scaler=None,
    use_amp=False,
    grad_clip=1.0,
    progress_every=100,
):
    is_train = (
        optimizer is not None
    )

    model.train(
        is_train
    )

    loss_sum = 0.0
    sample_count = 0

    all_true = []
    all_pred = []

    W_sum = None
    W_count = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):
        batch = move_batch_to_device(
            batch,
            device,
        )

        labels = batch[
            "y"
        ].float().view(-1)

        B = labels.size(0)

        if is_train:
            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            is_train
        ):
            with torch.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                preds = model(
                    batch
                ).view(-1)

                loss = criterion(
                    preds,
                    labels,
                )

            if is_train:
                if (
                    scaler is not None
                    and use_amp
                ):
                    scaler.scale(
                        loss
                    ).backward()

                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                else:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip,
                    )

                    optimizer.step()

        loss_sum += (
            float(loss.item())
            * B
        )

        sample_count += B

        all_true.append(
            labels.detach().cpu().numpy()
        )

        all_pred.append(
            preds.detach().cpu().numpy()
        )

        if model.last_W is not None:
            W = model.last_W.detach()

            current_sum = W.sum(
                dim=0
            )

            if W_sum is None:
                W_sum = current_sum
            else:
                W_sum = (
                    W_sum
                    + current_sum
                )

            W_count += W.size(0)

        if (
            progress_every > 0
            and (
                batch_idx % progress_every == 0
                or batch_idx == len(loader)
            )
        ):
            elapsed = time.time() - start_time

            print(
                f"    batch "
                f"{batch_idx}/{len(loader)} | "
                f"elapsed={elapsed / 60:.1f} min",
                flush=True,
            )

    y_true = np.concatenate(
        all_true,
        axis=0,
    )

    y_pred = np.concatenate(
        all_pred,
        axis=0,
    )

    metrics = regression_sentiment_metrics(
        y_true,
        y_pred,
    )

    metrics["Loss"] = (
        loss_sum
        / max(sample_count, 1)
    )

    metrics["Seconds"] = (
        time.time()
        - start_time
    )

    if (
        W_sum is not None
        and W_count > 0
    ):
        metrics["VoterWeights"] = (
            W_sum
            / W_count
        ).detach().cpu().numpy()
    else:
        metrics["VoterWeights"] = None

    return (
        metrics,
        y_true,
        y_pred,
    )


def format_metrics(metrics):
    parts = [
        f"loss={metrics['Loss']:.4f}",
        f"MAE={metrics['MAE']:.4f}",
        f"Corr={metrics['Corr']:.4f}",
        f"Acc7={metrics['Acc7']:.4f}",
        f"Has0-A2={metrics['Has0_Acc2']:.4f}",
        f"Has0-F1={metrics['Has0_F1']:.4f}",
        f"Non0-A2={metrics['Non0_Acc2']:.4f}",
        f"Non0-F1={metrics['Non0_F1']:.4f}",
    ]

    W = metrics.get(
        "VoterWeights"
    )

    if W is not None:
        parts.append(
            "W(t/v/a)="
            f"{W[0]:.3f}/"
            f"{W[1]:.3f}/"
            f"{W[2]:.3f}"
        )

    return " | ".join(parts)


def print_final_metrics(
    name,
    metrics,
):
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"N               : "
        f"{metrics['N']}"
    )

    print(
        f"MAE ↓           : "
        f"{metrics['MAE']:.6f}"
    )

    print(
        f"Corr ↑          : "
        f"{metrics['Corr']:.6f}"
    )

    print(
        f"Acc-7 ↑         : "
        f"{metrics['Acc7']:.6f}"
    )

    print(
        "Acc-2 Has0/Non0 : "
        f"{metrics['Has0_Acc2']:.6f} / "
        f"{metrics['Non0_Acc2']:.6f}"
    )

    print(
        "F1 Has0/Non0    : "
        f"{metrics['Has0_F1']:.6f} / "
        f"{metrics['Non0_F1']:.6f}"
    )

    print(
        f"Non0 N          : "
        f"{metrics['Non0_N']}"
    )

    W = metrics.get(
        "VoterWeights"
    )

    if W is not None:
        print(
            "Voter W(t/v/a)  : "
            f"{W[0]:.6f} / "
            f"{W[1]:.6f} / "
            f"{W[2]:.6f}"
        )


# ============================================================
# 8. Save outputs
# ============================================================

def _jsonable_metrics(metrics):
    out = {}

    for key, value in metrics.items():
        if key == "VoterWeights":
            out[key] = (
                None
                if value is None
                else [
                    float(x)
                    for x in value
                ]
            )
        elif isinstance(
            value,
            (np.floating, np.integer),
        ):
            out[key] = value.item()
        elif isinstance(value, float):
            out[key] = (
                None
                if math.isnan(value)
                else value
            )
        else:
            out[key] = value

    return out


def save_predictions_csv(
    path,
    test_data,
    y_true,
    y_pred,
):
    if len(test_data) != len(y_true):
        raise ValueError(
            "test_data and predictions length mismatch."
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "video_id",
                "clip_id",
                "label_name",
                "y_true",
                "y_pred",
                "true_acc7_class",
                "pred_acc7_class",
                "has0_true",
                "has0_pred",
                "is_non0",
                "non0_true",
                "non0_pred",
            ]
        )

        for item, yt, yp in zip(
            test_data,
            y_true,
            y_pred,
        ):
            yt = float(yt)
            yp = float(yp)

            is_non0 = (
                yt != 0.0
            )

            writer.writerow(
                [
                    item.get("video_id", ""),
                    item.get("clip_id", ""),
                    item.get("label", ""),
                    yt,
                    yp,
                    int(
                        np.round(
                            np.clip(
                                yt,
                                -3,
                                3,
                            )
                        )
                    ),
                    int(
                        np.round(
                            np.clip(
                                yp,
                                -3,
                                3,
                            )
                        )
                    ),
                    int(yt >= 0),
                    int(yp >= 0),
                    int(is_non0),
                    (
                        int(yt > 0)
                        if is_non0
                        else ""
                    ),
                    (
                        int(yp > 0)
                        if is_non0
                        else ""
                    ),
                ]
            )


# ============================================================
# 9. Main
# ============================================================

def train_and_test(args):
    set_seed(
        args.seed
    )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    use_amp = (
        device.type == "cuda"
        and not args.no_amp
    )

    print("=" * 80)
    print("MOSEI CONCAT unified regression")
    print("=" * 80)

    print("Device:", device)
    print("Seed:", args.seed)
    print("AMP:", use_amp)

    (
        train_data,
        valid_data,
        test_data,
        train_pack,
        valid_pack,
        test_pack,
    ) = build_regression_packs(
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        inspect_samples=args.inspect_samples,
        run_checks=not args.skip_checks,
    )

    (
        train_loader,
        valid_loader,
        test_loader,
    ) = build_loaders(
        train_pack,
        valid_pack,
        test_pack,
        train_data,
        valid_data,
        test_data,
        C=args.context_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = (
        ContextEmotionTriModalLLMRegressor(
            d_model=args.d_model,
            dropout=args.dropout,
            ctx_layers=args.ctx_layers,
            ctx_heads=args.ctx_heads,
            audio_in_dim=1024,
            ds_kernel=args.ds_kernel,
            ds_stride=args.ds_stride,
            voter_temperature=args.voter_temperature,
        )
        .to(device)
    )

    criterion = nn.L1Loss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(
        enabled=use_amp
    )

    checkpoint_path = os.path.join(
        args.output_dir,
        (
            f"best_MOSEI_MAE_"
            f"{args.seed}_"
            f"concat_regression.pth"
        ),
    )

    history = []
    best_valid_mae = float("inf")

    print(
        "\nBest checkpoint:",
        checkpoint_path,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        print(
            "\n" + "=" * 80
        )

        print(
            f"Epoch {epoch}/{args.epochs}"
        )

        print("=" * 80)
        print("[Train]")

        (
            train_metrics,
            _,
            _,
        ) = run_one_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            progress_every=args.progress_every,
        )

        print(
            "Train:",
            format_metrics(
                train_metrics
            ),
        )

        print("[Valid]")

        with torch.no_grad():
            (
                valid_metrics,
                _,
                _,
            ) = run_one_epoch(
                model,
                valid_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
                grad_clip=args.grad_clip,
                progress_every=0,
            )

        print(
            "Valid:",
            format_metrics(
                valid_metrics
            ),
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train":
                    _jsonable_metrics(
                        train_metrics
                    ),

                "valid":
                    _jsonable_metrics(
                        valid_metrics
                    ),
            }
        )

        if (
            valid_metrics["MAE"]
            < best_valid_mae
        ):
            best_valid_mae = (
                valid_metrics["MAE"]
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "best_valid_mae":
                        best_valid_mae,

                    "args":
                        vars(args),
                },
                checkpoint_path,
            )

            print(
                "✅ saved best-MAE checkpoint"
            )

    # ========================================================
    # Load best validation-MAE checkpoint
    # ========================================================

    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    print(
        "\nLoaded best checkpoint:",
        f"epoch={ckpt['epoch']} | "
        f"valid_MAE={ckpt['best_valid_mae']:.6f}"
    )

    # ========================================================
    # Test
    # ========================================================

    with torch.no_grad():
        (
            test_metrics,
            y_true,
            y_pred,
        ) = run_one_epoch(
            model,
            test_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            progress_every=0,
        )

    print_final_metrics(
        "MOSEI TEST RESULTS",
        test_metrics,
    )

    # ========================================================
    # Save history / result / predictions
    # ========================================================

    history_path = os.path.join(
        args.output_dir,
        (
            f"history_MOSEI_"
            f"{args.seed}_"
            f"concat_regression.json"
        ),
    )

    result_path = os.path.join(
        args.output_dir,
        (
            f"result_MOSEI_"
            f"{args.seed}_"
            f"concat_regression.json"
        ),
    )

    prediction_path = os.path.join(
        args.output_dir,
        (
            f"predictions_MOSEI_"
            f"{args.seed}_"
            f"concat_regression.csv"
        ),
    )

    with open(
        history_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        result_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "best_epoch":
                    int(ckpt["epoch"]),

                "best_valid_mae":
                    float(
                        ckpt[
                            "best_valid_mae"
                        ]
                    ),

                "test":
                    _jsonable_metrics(
                        test_metrics
                    ),

                "config":
                    vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    save_predictions_csv(
        prediction_path,
        test_data,
        y_true,
        y_pred,
    )

    print("\nSaved:")
    print("  checkpoint :", checkpoint_path)
    print("  history    :", history_path)
    print("  result     :", result_path)
    print("  predictions:", prediction_path)


# ============================================================
# 10. CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "MOSEI six-route CONCAT unified regression: "
            "MAE/Corr/Acc7/Has0 Acc2/Non0 Acc2."
        )
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
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--context-size",
        type=int,
        default=DEFAULT_CONTEXT_SIZE,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=DEFAULT_D_MODEL,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=DEFAULT_DROPOUT,
    )

    parser.add_argument(
        "--ctx-layers",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--ctx-heads",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--ds-kernel",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--ds-stride",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--voter-temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Default 0 because MOSEI raw-audio batches are very large "
            "and multiprocessing workers may exceed shared memory."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N training batches.",
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

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train_and_test(
        args
    )


if __name__ == "__main__":
    main()
