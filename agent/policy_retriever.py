"""Retrieval over northstar-policy-docs/ for the query_policy tool.

Chunking: one chunk per markdown section (## or ###), prefixed with the
document ID and heading path. Policy sections are short (a few hundred tokens)
and self-contained, and a clause like "this section overrides all other
sections" lives in the same section as the rule it overrides, so splitting
inside a section would separate rules from their exceptions. Fixed-size
chunking with overlap would cut across those boundaries; section chunking
keeps each rule and its exceptions together.

Embeddings: Titan Text Embeddings v2 (1024-d, normalized). Vector store: an
in-memory numpy matrix cached to disk (about 50 chunks, so FAISS adds nothing).
"""
import json
import re
from pathlib import Path

import boto3
import numpy as np

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
DOCS_DIR = Path(__file__).resolve().parent.parent / "northstar-policy-docs"
CACHE = Path(__file__).resolve().parent / ".cache" / "policy_index.npz"


def chunk_policy_docs(docs_dir: Path = DOCS_DIR) -> list[dict]:
    chunks = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        m = re.search(r"\*\*Document ID:\*\*\s*(\S+)", text)
        doc_id = m.group(1) if m else path.stem
        title = text.splitlines()[0].lstrip("# ").strip()
        heading, buf = "Overview", []

        def flush():
            body = "\n".join(buf).strip()
            if len(body) > 40:
                chunks.append({
                    "doc_id": doc_id,
                    "source": path.name,
                    "heading": heading,
                    "text": f"[{doc_id} | {title} | {heading}]\n{body}",
                })

        for line in text.splitlines()[1:]:
            h = re.match(r"^(#{2,3})\s+(.*)", line)
            if h:
                flush()
                heading, buf = h.group(2).strip(), []
            elif line.strip() != "---":
                buf.append(line)
        flush()
    return chunks


class PolicyRetriever:
    def __init__(self, region: str = "us-east-1", top_k: int = 4):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.top_k = top_k
        self.chunks = chunk_policy_docs()
        self.matrix = self._load_or_build()

    def _embed(self, text: str) -> np.ndarray:
        resp = self.client.invoke_model(
            modelId=EMBED_MODEL_ID,
            body=json.dumps({"inputText": text[:8000], "dimensions": 1024, "normalize": True}),
        )
        return np.array(json.loads(resp["body"].read())["embedding"], dtype="float32")

    def _load_or_build(self) -> np.ndarray:
        fingerprint = str(sum(len(c["text"]) for c in self.chunks)) + f":{len(self.chunks)}"
        if CACHE.exists():
            data = np.load(CACHE, allow_pickle=False)
            if str(data["fingerprint"]) == fingerprint:
                return data["matrix"]
        matrix = np.stack([self._embed(c["text"]) for c in self.chunks])
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, matrix=matrix, fingerprint=fingerprint)
        return matrix

    def search(self, question: str) -> list[dict]:
        scores = self.matrix @ self._embed(question)   # cosine: vectors are normalized
        top = np.argsort(-scores)[: self.top_k]
        return [{**self.chunks[i], "score": round(float(scores[i]), 3)} for i in top]
