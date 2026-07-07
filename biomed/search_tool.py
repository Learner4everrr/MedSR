import requests

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class BiomedSearchTool(BaseTool):
    """Small wrapper around the existing Search-R1 retriever HTTP API."""

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search biomedical documents for evidence.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Biomedical search query.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            }
        )

    async def execute(self, instance_id, parameters, **kwargs):
        query = str(parameters.get("query", "")).strip()
        if not query:
            return ToolResponse(text="Empty query."), 0.0, {"search/empty_query": 1}

        url = self.config.get("url", "http://127.0.0.1:8000/retrieve")
        topk = int(self.config.get("topk", 2))
        timeout = float(self.config.get("timeout", 30))
        max_doc_chars = int(self.config.get("max_doc_chars", 900))

        if topk <= 0:
            return ToolResponse(text="Retrieval disabled."), 0.0, {"search/disabled": 1}

        response = requests.post(
            url,
            json={"queries": [query], "topk": topk, "return_scores": True},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()

        docs = payload.get("result", payload)
        if isinstance(docs, dict):
            docs = docs.get("results", docs.get("documents", docs))
        if isinstance(docs, list) and docs and isinstance(docs[0], list):
            docs = docs[0]

        lines = []
        for i, doc in enumerate(docs[:topk], start=1):
            if isinstance(doc, dict):
                title = doc.get("title") or doc.get("id") or f"Doc {i}"
                text = doc.get("contents") or doc.get("content") or doc.get("text") or str(doc)
                score = doc.get("score", doc.get("retrieval_score", ""))
                prefix = f"Doc {i}"
                if title:
                    prefix += f" ({title})"
                if score != "":
                    prefix += f" [score={score}]"
                lines.append(f"{prefix}: {str(text)[:max_doc_chars]}")
            else:
                lines.append(f"Doc {i}: {str(doc)[:max_doc_chars]}")

        return ToolResponse(text="\n".join(lines) if lines else "No results."), 0.0, {"search/calls": 1}
