#!/bin/bash

# ============================================================
# finetune.py 支持两种微调方式：
#   - use_lora=True  => LoRA 微调 (默认)
#   - use_lora=False => 全量微调
# ============================================================

# === 模式 1: LoRA 微调 ===
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b" \
    --data_root_dir /mnt/lx/cyx/openvla/modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --use_lora True \
    --lora_rank 32 \
    --compute_importance True \
    --importance_compute_steps 2000 \
    --batch_size 8 \
    --grad_accumulation_steps 2 \
    --max_steps 50000 \
    --lora_importance_reduction sum

# === 模式 2: 全量微调 ===
torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
    --vla_path "openvla/openvla-7b-finetuned-libero-spatial" \
    --data_root_dir /mnt/lx/cyx/openvla/modified_libero_rlds \
    --dataset_name libero_spatial_no_noops \
    --use_lora False \
    --compute_importance True \
    --batch_size 4 \
    --grad_accumulation_steps 4 \
    --max_steps 50000 \
    --importance_compute_steps 100