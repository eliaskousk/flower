"""flowertune-medical-fdlora: A Flower / FlowerTune app using FDLoRA.

FDLoRA Strategy implementation following the paper's server-side aggregation.

The FDLoRA paper uses Nesterov momentum for the outer optimization on the server:
Δ^(t) = (1/N)∑(θₛ^(t-1) - θₛ^(i)(t))  # Averaged gradients
θₛ^(t) = θₛ^(t-1) - η * (μ * v^(t-1) + Δ^(t))  # Momentum update
"""

from collections.abc import Iterable
from logging import INFO, WARN
from typing import Optional

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


class FDLoRAStrategy(FedAvg):
    """FDLoRA strategy with Nesterov momentum-based server optimization.

    This strategy implements the FDLoRA server-side aggregation:
    1. Collect global LoRA updates from clients
    2. Compute averaged gradients (difference from previous global params)
    3. Apply Nesterov momentum update to global parameters
    """

    def __init__(
        self,
        server_momentum: float = 0.9,
        server_lr: float = 1.0,
        **kwargs
    ):
        """Initialize FDLoRA strategy.

        Args:
            server_momentum: Momentum coefficient for Nesterov SGD (default: 0.9)
            server_lr: Server learning rate (default: 1.0)
            **kwargs: Additional arguments passed to FedAvg
        """
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()
        self.server_momentum = server_momentum
        self.server_lr = server_lr
        self.velocity = None  # Momentum buffer
        self.prev_global_params = None  # Previous round's global parameters

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        messages = super().configure_train(server_round, arrays, config, grid)

        # Store current global params for gradient computation
        self.prev_global_params = {k: v.clone() for k, v in arrays.items()}

        # Track communication costs
        self.comm_tracker.track(messages)

        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords using FDLoRA's Nesterov momentum optimization.

        Instead of simple averaging, we compute:
        1. Average of client updates (gradients)
        2. Apply Nesterov momentum
        3. Update global parameters
        """
        # Track communication costs
        replies_list = list(replies)
        self.comm_tracker.track(replies_list)

        # Use parent's aggregation to get averaged parameters
        arrays, metrics = super().aggregate_train(server_round, iter(replies_list))

        if arrays is None or self.prev_global_params is None:
            return arrays, metrics

        # Convert to torch tensors for momentum computation
        current_params = arrays.to_torch_state_dict()

        # Compute averaged gradients: Δ = prev_params - averaged_client_params
        # (Note: client params after training are closer to optimal,
        # so gradient points from prev to current)
        gradients = {}
        for key in current_params:
            if key in self.prev_global_params:
                # Gradient = how much clients moved from previous global
                gradients[key] = self.prev_global_params[key] - current_params[key]

        # Initialize velocity if first round
        if self.velocity is None:
            self.velocity = {k: torch.zeros_like(v) for k, v in gradients.items()}

        # Nesterov momentum update
        # v^(t) = μ * v^(t-1) + Δ^(t)
        # θ^(t) = θ^(t-1) - η * v^(t)
        updated_params = {}
        for key in current_params:
            if key in gradients:
                # Update velocity with momentum
                self.velocity[key] = (
                    self.server_momentum * self.velocity[key] + gradients[key]
                )

                # Update parameters using Nesterov momentum
                # Nesterov lookahead: use velocity at next step
                updated_params[key] = (
                    self.prev_global_params[key] -
                    self.server_lr * (
                        self.server_momentum * self.velocity[key] + gradients[key]
                    )
                )
            else:
                updated_params[key] = current_params[key]

        # Create new ArrayRecord with momentum-updated parameters
        return ArrayRecord(updated_params), metrics


class CommunicationTracker:
    """Communication costs tracker over FL rounds."""

    def __init__(self):
        self.curr_comm_cost = 0.0

    def track(self, messages: Iterable[Message]):
        comm_cost = (
            sum(
                record.count_bytes()
                for msg in messages
                if msg.has_content()
                for record in msg.content.array_records.values()
            )
            / 1024**2
        )

        self.curr_comm_cost += comm_cost
        log(
            INFO,
            "Communication budget: used %.2f MB (+%.2f MB this round) / 200,000 MB",
            self.curr_comm_cost,
            comm_cost,
        )

        if self.curr_comm_cost > 2e5:
            log(
                WARN,
                "The accumulated communication cost has exceeded 200,000 MB. "
                "Please consider reducing it if you plan to participate "
                "FlowerTune LLM Leaderboard.",
            )
