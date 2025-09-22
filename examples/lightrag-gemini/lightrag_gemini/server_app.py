"""LightRAG Gemini: Flower ServerApp."""

import os
import asyncio
from time import sleep
from typing import List, Set

from flwr.common import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.server import Grid, ServerApp

from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.google import google_complete, google_embed
from lightrag.utils import GemmaTokenizer, logger

from .data import sample_texts


def load_sample_documents(num_clients: int, docs_per_client: int) -> List[List[str]]:
    """Load and distribute sample documents across clients."""
    
    # Distribute documents across clients
    client_documents = []
    for i in range(num_clients):
        start_idx = i * docs_per_client
        end_idx = start_idx + docs_per_client
        # Wrap around if we run out of documents
        docs = []
        for j in range(start_idx, end_idx):
            docs.append(sample_texts[j % len(sample_texts)])
        client_documents.append(docs)
    
    return client_documents


async def wait_for_nodes(grid: Grid, num_nodes: int) -> Set[int]:
    """Wait for required number of nodes to be online."""
    node_ids = {}
    while len(node_ids) < num_nodes:
        node_ids = grid.get_node_ids()
        if len(node_ids) < num_nodes:
            sleep(1)
    return node_ids


async def aggregate_responses(question: str, responses: List[RecordDict]) -> str:
    """Aggregate responses from multiple clients using LightRAG."""
    # Combine all responses
    combined_text = "\n\n".join([
        f"Client {r['result']['node_id']} ({r['result']['mode']} mode):\n{r['result']['response']}"
        for r in responses
    ])

    # logger.info(f"Combined responses:\n{combined_text}")
    
    # Use a server-side LightRAG instance to synthesize the final answer
    model = os.environ["LLM_MODEL"] if "LLM_MODEL" in os.environ else "gemini-2.5-flash-lite"
    
    server_rag = LightRAG(
        working_dir="./lightrag_data/server",
        llm_model_name=model,
        llm_model_func=google_complete,
        tokenizer=GemmaTokenizer(),
        embedding_func=google_embed,
    )

    await server_rag.initialize_storages()
    await initialize_pipeline_status()
    logger.info(f"ServerApp: LightRAG initialized")

    await server_rag.ainsert(combined_text)
    
    # Create a synthesis prompt
    synthesis_prompt = f"""Your knowledge base has responses from multiple federated clients.
    Provide a comprehensive and synthesized answer for the following question: {question}"""
    
    final_answer = await server_rag.aquery(synthesis_prompt, param=QueryParam(mode="hybrid"))
    await server_rag.finalize_storages()
    
    return final_answer


async def init(grid: Grid, node_ids: set[int], config: ConfigRecord):
    # Initialize LightRAG on all clients
    logger.info("Initializing LightRAG on clients...")
    messages = []
    for node_id in node_ids:
        msg = Message(
            content=RecordDict({"config": config}),
            message_type=MessageType.QUERY + ".init",
            dst_node_id=node_id,
            group_id="init",
        )
        messages.append(msg)

    replies = grid.send_and_receive(messages, timeout=60)
    logger.info(f"Initialized {len(replies)}/{len(messages)} LightRAG instances")


async def insert(grid: Grid, node_ids: set[int], num_clients: int, docs_per_client: int):
    # Load and distribute documents
    logger.info("Loading and distributing documents...")
    client_documents = load_sample_documents(num_clients, docs_per_client)

    # Insert documents into each client's LightRAG
    messages = []
    for node_id, docs in zip(node_ids, client_documents):
        config_record = ConfigRecord({"documents": docs})
        msg = Message(
            content=RecordDict({"config": config_record}),
            message_type=MessageType.QUERY + ".insert",
            dst_node_id=node_id,
            group_id="insert",
        )
        messages.append(msg)

    replies = grid.send_and_receive(messages, timeout=300)
    total_inserted = sum(r.content["status"]["inserted"] for r in replies)
    logger.info(f"Inserted {total_inserted} documents across {len(replies)}/{len(messages)} clients")


async def query(grid: Grid, node_ids: set[int], query_mode: int, top_k: int):
    # Example queries to test the federated RAG system
    test_queries = [
        "What are the main technological advances discussed?",
        "How is AI being used in different fields?",
        "What are the environmental and sustainability topics mentioned?",
        "What are the future technologies that will transform society?",
    ]

    for query_idx, question in enumerate(test_queries):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Query {query_idx + 1}: {question}")
        logger.info(f"{'=' * 60}")

        # Send query to all clients
        messages = []
        for node_id in node_ids:
            config_record = ConfigRecord({
                "question": question,
                "mode": query_mode,
                "top_k": top_k,
            })
            msg = Message(
                content=RecordDict({"config": config_record}),
                message_type=MessageType.QUERY + ".query",
                dst_node_id=node_id,
                group_id=f"query_{query_idx}",
            )
            messages.append(msg)

        replies = grid.send_and_receive(messages, timeout=120)

        # Extract results
        results = [r.content for r in replies]

        # Aggregate responses
        logger.info("Aggregating responses from clients...")
        final_answer = await aggregate_responses(question, results)

        logger.info(f"\nFinal Answer:\n{final_answer}")


async def async_main(grid: Grid, context: Context) -> None:
    """Async main server logic."""

    # Get configuration
    # config = context.run_config
    config = ConfigRecord(context.run_config)
    num_clients = config.get("num-supernodes", 3)
    docs_per_client = config.get("docs-per-client", 10)
    query_mode = config.get("query-mode", "hybrid")
    top_k = config.get("top-k", 5)

    # Wait for clients
    logger.info(f"Waiting for {num_clients} clients...")
    node_ids = await wait_for_nodes(grid, num_clients)
    logger.info(f"Connected to {len(node_ids)} clients")

    await init(grid, node_ids, config)
    await insert(grid, node_ids, num_clients, docs_per_client)
    await query(grid, node_ids, query_mode, top_k)

    logger.info("\n" + "=" * 60)
    logger.info("Federated LightRAG example completed successfully!")

# Flower ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main server application."""
    asyncio.run(async_main(grid, context))
