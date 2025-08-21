"""huggingface_example: A Flower / Hugging Face app."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from huggingface_example.task import get_params, get_model


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate accuracy metrics using weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    # Set global model initialization
    model_name = context.run_config["model-name"]
    ndarrays = get_params(get_model(model_name))
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define strategy
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        initial_parameters=global_model_init,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
