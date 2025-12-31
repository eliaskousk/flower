"""flowertune-medical-fdlora: A Flower / FlowerTune app using FDLoRA.

This module implements FDLoRA (Federated Dual Low-Rank Adaptation) following
the paper: "FDLoRA: Personalized Federated Learning of Large Language Model
via Dual LoRA Tuning" (arXiv:2406.07925).

FDLoRA uses two LoRA modules per client:
- Global LoRA: Parameters shared across clients via federated aggregation
- Personal LoRA: Parameters kept local for client-specific knowledge
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from peft.tuners.lora import LoraLayer
from peft.utils import prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


class DualLoraLinear(nn.Module):
    """A linear layer with dual LoRA adapters for FDLoRA.

    This module wraps a base linear layer and adds two low-rank adapters:
    - Global LoRA (A_g, B_g): Aggregated across clients
    - Personal LoRA (A_p, B_p): Kept local per client

    The output combines both: h = W_0 * x + B_g @ A_g @ x + B_p @ A_p @ x
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Global LoRA parameters (will be federated)
        self.lora_A_global = nn.Linear(in_features, r, bias=False)
        self.lora_B_global = nn.Linear(r, out_features, bias=False)

        # Personal LoRA parameters (kept local)
        self.lora_A_personal = nn.Linear(in_features, r, bias=False)
        self.lora_B_personal = nn.Linear(r, out_features, bias=False)

        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Adaptive fusion weights (w1 for global, w2 for personal)
        # These are learned during the fusion stage
        self.fusion_weight_global = nn.Parameter(torch.tensor(0.5))
        self.fusion_weight_personal = nn.Parameter(torch.tensor(0.5))

        # Initialize LoRA weights
        self._init_weights()

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def _init_weights(self):
        """Initialize LoRA weights following standard practice."""
        # Initialize A with Kaiming uniform and B with zeros
        nn.init.kaiming_uniform_(self.lora_A_global.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_global.weight)
        nn.init.kaiming_uniform_(self.lora_A_personal.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_personal.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining base layer with dual LoRA outputs."""
        # Base model output
        result = self.base_layer(x)

        # Apply dropout
        x_dropped = self.lora_dropout(x)

        # Global LoRA contribution
        global_lora = self.lora_B_global(self.lora_A_global(x_dropped)) * self.scaling

        # Personal LoRA contribution
        personal_lora = self.lora_B_personal(self.lora_A_personal(x_dropped)) * self.scaling

        # Combine with fusion weights (normalized via softmax for stability)
        weights = torch.softmax(
            torch.stack([self.fusion_weight_global, self.fusion_weight_personal]), dim=0
        )
        result = result + weights[0] * global_lora + weights[1] * personal_lora

        return result


def get_model(model_cfg: DictConfig):
    """Load model with appropriate quantization config and dual LoRA for FDLoRA."""
    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    # Use standard PEFT LoRA config for the base structure
    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )

    if model_cfg.gradient_checkpointing:
        model.config.use_cache = False

    peft_model = get_peft_model(model, peft_config)

    return peft_model


def get_fdlora_model(model_cfg: DictConfig):
    """Load model with FDLoRA dual adapter structure.

    This creates a model with both global and personal LoRA adapters
    following the FDLoRA paper architecture.
    """
    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}/"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
    )

    # Create global LoRA adapter
    global_lora_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )

    # Create personal LoRA adapter with same config
    personal_lora_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )

    if model_cfg.gradient_checkpointing:
        model.config.use_cache = False

    # Apply global LoRA adapter
    peft_model = get_peft_model(model, global_lora_config, adapter_name="global")

    # Add personal LoRA adapter
    peft_model.add_adapter("personal", personal_lora_config)

    return peft_model


def get_global_lora_state_dict(model) -> dict:
    """Extract only the global LoRA parameters for federation."""
    model.set_adapter("global")
    return get_peft_model_state_dict(model, adapter_name="global")


def get_personal_lora_state_dict(model) -> dict:
    """Extract only the personal LoRA parameters."""
    model.set_adapter("personal")
    return get_peft_model_state_dict(model, adapter_name="personal")


def get_all_lora_state_dict(model) -> dict:
    """Extract both global and personal LoRA parameters."""
    global_state = get_peft_model_state_dict(model, adapter_name="global")
    personal_state = get_
    peft_model_state_dict(model, adapter_name="personal")

    # Prefix keys to distinguish them
    all_state = {}
    for k, v in global_state.items():
        all_state[f"global.{k}"] = v
    for k, v in personal_state.items():
        all_state[f"personal.{k}"] = v

    return all_state


def set_global_lora_state_dict(model, state_dict: dict):
    """Set the global LoRA parameters from a state dict."""
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, state_dict, adapter_name="global")


def set_personal_lora_state_dict(model, state_dict: dict):
    """Set the personal LoRA parameters from a state dict."""
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, state_dict, adapter_name="personal")


def enable_both_adapters(model):
    """Enable both global and personal adapters for combined inference."""
    model.set_adapter(["global", "personal"])


def enable_global_adapter_only(model):
    """Enable only the global adapter."""
    model.set_adapter("global")


def enable_personal_adapter_only(model):
    """Enable only the personal adapter."""
    model.set_adapter("personal")


def sync_personal_from_global(model, sync_ratio: float = 1.0):
    """Synchronize personal LoRA parameters from global LoRA.

    This implements the periodic synchronization mentioned in the FDLoRA paper,
    where personal LoRA can optionally sync with global LoRA every H rounds.

    Args:
        model: The PEFT model with dual adapters
        sync_ratio: How much to blend (1.0 = full copy, 0.5 = average)
    """
    global_state = get_peft_model_state_dict(model, adapter_name="global")
    personal_state = get_peft_model_state_dict(model, adapter_name="personal")

    if sync_ratio == 1.0:
        set_personal_lora_state_dict(model, global_state)
    else:
        # Blend personal and global
        blended_state = {}
        for key in global_state:
            if key in personal_state:
                blended_state[key] = (
                    sync_ratio * global_state[key] +
                    (1 - sync_ratio) * personal_state[key]
                )
            else:
                blended_state[key] = global_state[key]
        set_personal_lora_state_dict(model, blended_state)
