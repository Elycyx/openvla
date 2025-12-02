"""
adam_importance.py

Compute parameter importance based on Adam optimizer states.
Uses bias-corrected first and second moment estimates to calculate:
    I_k = |m̂_k * w_k - 0.5 * v̂_k * w_k²|

Supports block-wise aggregation for LLM layers (attention and MLP blocks).
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


# === LLM Layer Structure Patterns ===
# Matches patterns like: layers.0.self_attn.q_proj, model.layers.15.mlp.gate_proj
ATTENTION_PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]

# Pattern to extract layer index
LAYER_PATTERN = re.compile(r"layers\.(\d+)\.")


class AdamImportance:
    """
    Compute parameter importance using Adam optimizer's moment estimates.
    
    Args:
        beta1: Adam's first moment decay rate (default: 0.9)
        beta2: Adam's second moment decay rate (default: 0.999)
    """
    
    def __init__(self, beta1: float = 0.9, beta2: float = 0.999):
        self.beta1 = beta1
        self.beta2 = beta2
    
    @torch.no_grad()
    def compute_param_importance_sum(
        self,
        param: torch.Tensor,
        optimizer: Optimizer,
        param_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> float:
        """
        Compute importance sum for a single parameter using Adam's moment estimates.
        
        I_k = |m̂_k * w_k - 0.5 * v̂_k * w_k²|
        
        Returns the sum of importance values as a scalar to avoid GPU memory overhead.
        
        Args:
            param: The parameter tensor
            optimizer: The optimizer containing the state
            param_state: Optional pre-fetched optimizer state for this parameter
            
        Returns:
            Sum of importance values (scalar float)
        """
        if param_state is None:
            param_state = optimizer.state.get(param, {})
        
        if not param_state or "exp_avg" not in param_state:
            # No optimizer state yet, return magnitude-based importance
            return param.data.abs().sum().item()
        
        # Directly use optimizer states (no extra memory allocation)
        m = param_state["exp_avg"]  # First moment (mean of gradients)
        v = param_state["exp_avg_sq"]  # Second moment (mean of squared gradients)
        step = param_state.get("step", 1)
        
        # Handle step as tensor (PyTorch >= 2.0) or int
        if isinstance(step, torch.Tensor):
            step = step.item()
        step = max(step, 1)  # Avoid division by zero
        
        # Bias correction factors (scalars, no GPU memory)
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        
        # Compute importance sum directly without storing intermediate tensors
        # I_k = |m̂_k * w_k - 0.5 * v̂_k * w_k²|
        # = |(m/bc1) * w - 0.5 * (v/bc2) * w²|
        w = param.data
        importance_sum = torch.abs(
            (m / bias_correction1) * w - 0.5 * (v / bias_correction2) * w.pow(2)
        ).sum().item()
        
        return importance_sum

    @torch.no_grad()
    def compute_param_importance_stats(
        self,
        param: torch.Tensor,
        optimizer: Optimizer,
        param_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[float, float]:
        """
        Compute importance statistics (sum and mean) for a single parameter.
        
        Returns both sum and mean to support different reduction methods.
        
        Args:
            param: The parameter tensor
            optimizer: The optimizer containing the state
            param_state: Optional pre-fetched optimizer state for this parameter
            
        Returns:
            Tuple of (sum, mean) of importance values
        """
        if param_state is None:
            param_state = optimizer.state.get(param, {})
        
        if not param_state or "exp_avg" not in param_state:
            # No optimizer state yet, return magnitude-based importance
            abs_data = param.data.abs()
            return abs_data.sum().item(), abs_data.mean().item()
        
        m = param_state["exp_avg"]
        v = param_state["exp_avg_sq"]
        step = param_state.get("step", 1)
        
        if isinstance(step, torch.Tensor):
            step = step.item()
        step = max(step, 1)
        
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        
        w = param.data
        importance = torch.abs(
            (m / bias_correction1) * w - 0.5 * (v / bias_correction2) * w.pow(2)
        )
        
        return importance.sum().item(), importance.mean().item()


def _get_layer_index(name: str) -> Optional[int]:
    """Extract layer index from parameter name."""
    match = LAYER_PATTERN.search(name)
    if match:
        return int(match.group(1))
    return None


def _get_proj_type(name: str) -> Optional[str]:
    """Get the projection type (q_proj, k_proj, etc.) from parameter name."""
    for proj_name in ATTENTION_PROJ_NAMES + MLP_PROJ_NAMES:
        if proj_name in name:
            return proj_name
    return None


def _is_llm_param(name: str) -> bool:
    """Check if parameter belongs to LLM backbone."""
    # Exclude vision_backbone and projector
    if "vision_backbone" in name or "featurizer" in name:
        return False
    # Check for LLM layer patterns
    if "layers." in name and any(proj in name for proj in ATTENTION_PROJ_NAMES + MLP_PROJ_NAMES):
        return True
    return False


def _aggregate_lora_importance(
    sum_a: float,
    mean_a: float,
    sum_b: float,
    mean_b: float,
    reduction: Literal["sum", "mean", "weighted"] = "sum",
) -> float:
    """
    Aggregate LoRA A and B matrix importance.
    
    Args:
        sum_a: Sum of importance for LoRA A matrix
        mean_a: Mean of importance for LoRA A matrix
        sum_b: Sum of importance for LoRA B matrix
        mean_b: Mean of importance for LoRA B matrix
        reduction: Reduction method - "sum", "mean", or "weighted"
        
    Returns:
        Aggregated importance value
    """
    if reduction == "sum":
        return sum_a + sum_b
    elif reduction == "mean":
        return mean_a + mean_b
    elif reduction == "weighted":
        return sum_a * 0.5 + sum_b * 0.5
    else:
        raise ValueError(f"Unknown reduction method: {reduction}")


@torch.no_grad()
def compute_importance_full(
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    run_dir: Path,
    beta1: float = 0.9,
    beta2: float = 0.999,
    save_json: bool = True,
    save_plot: bool = True,
) -> Dict[str, Any]:
    """
    Compute importance for full fine-tuning (all parameters).
    Only computes importance for LLM layers, not vision backbone.
    
    Args:
        model: The model (can be wrapped in DDP)
        optimizer: The AdamW optimizer
        step: Current training step
        run_dir: Directory to save results
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        save_json: Whether to save JSON file
        save_plot: Whether to save visualization plots
        
    Returns:
        Dictionary with importance data
    """
    adam_imp = AdamImportance(beta1=beta1, beta2=beta2)
    
    # Unwrap DDP if needed
    if hasattr(model, "module"):
        model_unwrapped = model.module
    else:
        model_unwrapped = model
    
    # Collect importance by layer and projection type
    layer_importance: Dict[int, Dict[str, Dict[str, float]]] = {}
    
    for name, param in model_unwrapped.named_parameters():
        if not param.requires_grad:
            continue
        if not _is_llm_param(name):
            continue
        
        layer_idx = _get_layer_index(name)
        proj_type = _get_proj_type(name)
        
        if layer_idx is None or proj_type is None:
            continue
        
        # Compute importance (returns scalar directly, no GPU memory overhead)
        imp_value = adam_imp.compute_param_importance_sum(param, optimizer)
        
        # Initialize layer dict if needed
        if layer_idx not in layer_importance:
            layer_importance[layer_idx] = {
                "attention": {"importance": 0.0},
                "mlp": {"importance": 0.0},
            }
        
        # Aggregate by block type
        if proj_type in ATTENTION_PROJ_NAMES:
            layer_importance[layer_idx]["attention"][proj_type] = imp_value
            layer_importance[layer_idx]["attention"]["importance"] += imp_value
        elif proj_type in MLP_PROJ_NAMES:
            layer_importance[layer_idx]["mlp"][proj_type] = imp_value
            layer_importance[layer_idx]["mlp"]["importance"] += imp_value
    
    # Build result dictionary
    result = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "mode": "full_finetune",
        "config": {
            "beta1": beta1,
            "beta2": beta2,
        },
        "layers": {f"layer_{idx}": data for idx, data in sorted(layer_importance.items())},
    }
    
    # Save results
    if save_json or save_plot:
        importance_dir = run_dir / "importance"
        os.makedirs(importance_dir, exist_ok=True)
        
        if save_json:
            save_importance_json(result, importance_dir, step)
        
        if save_plot:
            plot_importance(result, importance_dir, step)
    
    return result


@torch.no_grad()
def compute_importance_lora(
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    run_dir: Path,
    reduction: Literal["sum", "mean", "weighted"] = "sum",
    beta1: float = 0.9,
    beta2: float = 0.999,
    save_json: bool = True,
    save_plot: bool = True,
) -> Dict[str, Any]:
    """
    Compute importance for LoRA fine-tuning.
    Uses LoRA A and B matrix importance to represent the corresponding Linear layer.
    
    Args:
        model: The model with LoRA adapters (can be wrapped in DDP)
        optimizer: The AdamW optimizer
        step: Current training step
        run_dir: Directory to save results
        reduction: LoRA importance reduction method - "sum", "mean", or "weighted"
        beta1: Adam beta1 parameter
        beta2: Adam beta2 parameter
        save_json: Whether to save JSON file
        save_plot: Whether to save visualization plots
        
    Returns:
        Dictionary with importance data
    """
    adam_imp = AdamImportance(beta1=beta1, beta2=beta2)
    
    # Unwrap DDP if needed
    if hasattr(model, "module"):
        model_unwrapped = model.module
    else:
        model_unwrapped = model
    
    # Collect LoRA parameters grouped by their base layer
    # Key: (layer_idx, proj_type), Value: {"lora_A": param, "lora_B": param}
    lora_params: Dict[Tuple[int, str], Dict[str, Tuple[str, nn.Parameter]]] = {}
    
    for name, param in model_unwrapped.named_parameters():
        if not param.requires_grad:
            continue
        
        # Check if this is a LoRA parameter
        if "lora_A" not in name and "lora_B" not in name:
            continue
        
        # Skip non-LLM parameters
        if not _is_llm_param(name):
            continue
        
        layer_idx = _get_layer_index(name)
        proj_type = _get_proj_type(name)
        
        if layer_idx is None or proj_type is None:
            continue
        
        key = (layer_idx, proj_type)
        if key not in lora_params:
            lora_params[key] = {}
        
        if "lora_A" in name:
            lora_params[key]["lora_A"] = (name, param)
        elif "lora_B" in name:
            lora_params[key]["lora_B"] = (name, param)
    
    # Compute importance for each layer/projection
    layer_importance: Dict[int, Dict[str, Dict[str, float]]] = {}
    
    for (layer_idx, proj_type), lora_dict in lora_params.items():
        if "lora_A" not in lora_dict or "lora_B" not in lora_dict:
            continue
        
        _, param_a = lora_dict["lora_A"]
        _, param_b = lora_dict["lora_B"]
        
        # Compute importance stats for A and B matrices (returns scalar, no GPU overhead)
        sum_a, mean_a = adam_imp.compute_param_importance_stats(param_a, optimizer)
        sum_b, mean_b = adam_imp.compute_param_importance_stats(param_b, optimizer)
        
        # Aggregate using specified reduction method
        imp_value = _aggregate_lora_importance(sum_a, mean_a, sum_b, mean_b, reduction)
        
        # Initialize layer dict if needed
        if layer_idx not in layer_importance:
            layer_importance[layer_idx] = {
                "attention": {"importance": 0.0},
                "mlp": {"importance": 0.0},
            }
        
        # Aggregate by block type
        if proj_type in ATTENTION_PROJ_NAMES:
            layer_importance[layer_idx]["attention"][proj_type] = imp_value
            layer_importance[layer_idx]["attention"]["importance"] += imp_value
        elif proj_type in MLP_PROJ_NAMES:
            layer_importance[layer_idx]["mlp"][proj_type] = imp_value
            layer_importance[layer_idx]["mlp"]["importance"] += imp_value
    
    # Build result dictionary
    result = {
        "step": step,
        "timestamp": datetime.now().isoformat(),
        "mode": "lora_finetune",
        "config": {
            "beta1": beta1,
            "beta2": beta2,
            "lora_reduction": reduction,
        },
        "layers": {f"layer_{idx}": data for idx, data in sorted(layer_importance.items())},
    }
    
    # Save results
    if save_json or save_plot:
        importance_dir = run_dir / "importance"
        os.makedirs(importance_dir, exist_ok=True)
        
        if save_json:
            save_importance_json(result, importance_dir, step)
        
        if save_plot:
            plot_importance(result, importance_dir, step)
    
    return result


def save_importance_json(
    importance_data: Dict[str, Any],
    save_dir: Path,
    step: int,
) -> Path:
    """
    Save importance data to JSON file.
    
    Args:
        importance_data: Dictionary containing importance data
        save_dir: Directory to save the file
        step: Current training step
        
    Returns:
        Path to the saved file
    """
    save_path = save_dir / f"importance_step_{step:06d}.json"
    
    with open(save_path, "w") as f:
        json.dump(importance_data, f, indent=2)
    
    return save_path


def plot_importance(
    importance_data: Dict[str, Any],
    save_dir: Path,
    step: int,
) -> Tuple[Path, Path]:
    """
    Generate and save importance visualization plots.
    
    Creates:
    1. Heatmap: Layer × Block type importance matrix
    2. Bar chart: Importance comparison across layers
    
    Args:
        importance_data: Dictionary containing importance data
        save_dir: Directory to save the plots
        step: Current training step
        
    Returns:
        Tuple of paths to saved plots (heatmap_path, bar_path)
    """
    # Lazy import to avoid blocking in multi-process environment
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    layers_data = importance_data.get("layers", {})
    if not layers_data:
        return None, None
    
    # Extract data for plotting
    layer_names = sorted(layers_data.keys(), key=lambda x: int(x.split("_")[1]))
    attention_importance = []
    mlp_importance = []
    
    for layer_name in layer_names:
        layer_data = layers_data[layer_name]
        attention_importance.append(layer_data.get("attention", {}).get("importance", 0))
        mlp_importance.append(layer_data.get("mlp", {}).get("importance", 0))
    
    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # === Heatmap ===
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    
    heatmap_data = np.array([attention_importance, mlp_importance])
    
    sns.heatmap(
        heatmap_data,
        ax=ax1,
        cmap="YlOrRd",
        annot=False,
        xticklabels=[f"L{i}" for i in range(len(layer_names))],
        yticklabels=["Attention", "MLP"],
        cbar_kws={"label": "Importance"},
    )
    
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Block Type")
    ax1.set_title(f"Parameter Importance Heatmap (Step {step})")
    
    heatmap_path = save_dir / f"importance_heatmap_step_{step:06d}.png"
    fig1.tight_layout()
    fig1.savefig(heatmap_path, dpi=150)
    plt.close(fig1)
    
    # === Bar Chart ===
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(layer_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, attention_importance, width, label="Attention", color="#3498db")
    bars2 = ax2.bar(x + width/2, mlp_importance, width, label="MLP", color="#e74c3c")
    
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Importance")
    ax2.set_title(f"Parameter Importance by Layer (Step {step})")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"L{i}" for i in range(len(layer_names))], rotation=45, ha="right")
    ax2.legend()
    
    bar_path = save_dir / f"importance_bar_step_{step:06d}.png"
    fig2.tight_layout()
    fig2.savefig(bar_path, dpi=150)
    plt.close(fig2)
    
    return heatmap_path, bar_path

