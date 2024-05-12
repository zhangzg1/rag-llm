import logging
from typing import Dict, List, Any
from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from zhipuai import ZhipuAI
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("zhipu_api_key")


class ZhipuAIEmbeddings(BaseModel, Embeddings):

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not self.client:
            self.client = ZhipuAI(api_key=self.api_key)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        if "client" not in values or values["client"] is None:
            values["client"] = ZhipuAI(api_key=api_key)  # 确保传递了 api_key
        return values

    def _embed(self, texts: str) -> List[float]:
        embeddings = self.client.embeddings.create(
            model="embedding-2",
            input=texts
        )
        return embeddings.data[0].embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        resp = self.embed_documents([text])
        return resp[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError("Please use `embed_documents`. Official does not support asynchronous requests")

    async def aembed_query(self, text: str) -> List[float]:
        raise NotImplementedError("Please use `embed_query`. Official does not support asynchronous requests")
