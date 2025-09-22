import os
import shutil
from typing import Optional

from flwr.common import ConfigRecord

from lightrag import LightRAG, QueryParam
from lightrag.llm.google import google_complete, google_embed, google_embed_insert
from lightrag.utils import GemmaTokenizer, logger
from lightrag.kg.shared_storage import initialize_pipeline_status


class LightRAGClient:
    """Client wrapper for LightRAG instance."""

    def __init__(self, node_id: str, config: ConfigRecord):
        self.node_id = node_id
        self.config = config
        self.rag: Optional[LightRAG] = None
        self.working_dir = f"./lightrag_data/client_{node_id}"

        # Set up environment variables for VertexAI and Gemini models
        if "GOOGLE_GENAI_USE_VERTEXAI" not in os.environ:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = config.get("use-vertex-ai", "true")
        if "GOOGLE_CLOUD_LOCATION" not in os.environ:
            os.environ["GOOGLE_CLOUD_LOCATION"] = config.get("google-cloud-location", "us-central1")
        if "GOOGLE_CLOUD_PROJECT" not in os.environ:
            os.environ["GOOGLE_CLOUD_PROJECT"] = config.get("google-cloud-project", "your-project-id")
        if "LLM_MODEL" not in os.environ:
            os.environ["LLM_MODEL"] = config.get("llm-model", "gemini-2.5-flash-lite")
        if "EMBEDDING_MODEL" not in os.environ:
            os.environ["EMBEDDING_MODEL"] = config.get("embedding-model", "gemini-embedding-001")

        self.llm_model = os.environ["LLM_MODEL"]
        self.embedding_model = os.environ["EMBEDDING_MODEL"]

    async def initialize(self, task_type: Optional[str] = None, first: bool = True):
        """Initialize LightRAG instance."""
        # Clean up existing data for fresh start
        if first:
            if os.path.exists(self.working_dir):
                shutil.rmtree(self.working_dir)
            os.makedirs(self.working_dir, exist_ok=True)

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_name=self.llm_model,
            llm_model_func=google_complete,
            tokenizer=GemmaTokenizer(),
            embedding_func=google_embed_insert if task_type == "RETRIEVAL_DOCUMENT" else google_embed,
        )

        await self.rag.initialize_storages()
        await initialize_pipeline_status()
        logger.info(f"ClientApp {self.node_id}: LightRAG initialized")

    async def insert_documents(self, documents: list[str]):
        """Insert documents into LightRAG."""
        for idx, doc in enumerate(documents):
            await self.rag.ainsert(doc)
            logger.info(f"ClientApp {self.node_id}: Inserted document {idx + 1}/{len(documents)}")
        return len(documents)

    async def query(self, question: str, mode: str = "hybrid", top_k: int = 5) -> dict:
        """Query LightRAG instance."""
        result = await self.rag.aquery(
            question,
            param=QueryParam(mode=mode, top_k=top_k)
        )
        return {
            "node_id": self.node_id,
            "response": result,
            "mode": mode,
        }

    async def cleanup(self):
        """Clean up resources."""
        if self.rag:
            await self.rag.finalize_storages()
