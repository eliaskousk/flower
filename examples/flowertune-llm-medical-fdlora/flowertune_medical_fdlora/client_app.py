"""flowertune-medical-fdlora: A Flower / FlowerTune app using FDLoRA.

This client implements FDLoRA (Federated Dual Low-Rank Adaptation) which uses
two LoRA adapters per client:
- Global LoRA: Parameters aggregated across all clients via FedAvg
- Personal LoRA: Parameters kept local for personalized learning

Only global LoRA parameters are sent to/received from the server.
"""

import os
import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments
from trl import SFTTrainer

from flowertune_medical_fdlora.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
from flowertune_medical_fdlora.models import (
    cosine_annealing,
    get_fdlora_model,
    get_global_lora_state_dict,
    set_global_lora_state_dict,
    set_personal_lora_state_dict,
    sync_personal_from_global,
)

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# Store personal LoRA state per client (persisted across rounds)
PERSONAL_LORA_CACHE = {}


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data using FDLoRA.

    FDLoRA training procedure:
    1. Receive global LoRA parameters from server
    2. Load/restore personal LoRA parameters (kept local)
    3. Train both adapters jointly on local data
    4. Send only global LoRA parameters back to server
    5. Cache personal LoRA parameters locally for next round
    """
    # Parse config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    training_arguments = TrainingArguments(**cfg.train.training_arguments)

    # FDLoRA specific config
    sync_personal_every = cfg.fdlora.get("sync_personal_every", 0)  # 0 = never sync
    sync_ratio = cfg.fdlora.get("sync_ratio", 1.0)

    # Let's get the client partition
    trainset = load_data(partition_id, num_partitions, cfg.static.dataset.name)
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    # Load the FDLoRA model with dual adapters
    model = get_fdlora_model(cfg.model)

    # Set global LoRA from server
    global_state_dict = msg.content["arrays"].to_torch_state_dict()
    set_global_lora_state_dict(model, global_state_dict)

    # Restore or initialize personal LoRA
    server_round = msg.content["config"]["server-round"]
    if partition_id in PERSONAL_LORA_CACHE:
        # Restore personal LoRA from previous round
        set_personal_lora_state_dict(model, PERSONAL_LORA_CACHE[partition_id])
    else:
        # First round: initialize personal LoRA from global LoRA
        set_personal_lora_state_dict(model, global_state_dict)

    # Optional: Sync personal LoRA with global every H rounds
    if sync_personal_every > 0 and server_round % sync_personal_every == 0:
        sync_personal_from_global(model, sync_ratio=sync_ratio)

    # Enable both adapters for training
    model.set_adapter(["global", "personal"])

    # Set learning rate for current round
    new_lr = cosine_annealing(
        server_round,
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )

    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = msg.content["config"]["save_path"]

    # Construct trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=cfg.train.seq_length,
        train_dataset=trainset,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    # Do local training (both global and personal LoRA are trained)
    results = trainer.train()

    # Cache personal LoRA for next round (stays local)
    PERSONAL_LORA_CACHE[partition_id] = get_peft_model_state_dict(
        model, adapter_name="personal"
    )

    # Only send global LoRA parameters to server for aggregation
    global_lora_state = get_global_lora_state_dict(model)
    model_record = ArrayRecord(global_lora_state)

    metrics = {
        "train_loss": results.training_loss,
        "num-examples": len(trainset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
