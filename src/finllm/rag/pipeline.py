"""RAG pipeline: retrieve top-k chunks from Chroma, generate with the base model."""

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = """Answer the question using only the context below. If the context does not \
contain the answer, say so.

Context:
{context}

Question: {question}"""


class RAGPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.embedder = SentenceTransformer(
            config["embedding"]["model"], device=config["embedding"]["device"]
        )
        client = chromadb.PersistentClient(path=config["vectorstore"]["persist_dir"])
        self.collection = client.get_collection(config["vectorstore"]["collection"])

        model_name = config["model"]["base_model"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def retrieve(self, question: str) -> list[dict]:
        embedding = self.embedder.encode([question]).tolist()
        result = self.collection.query(
            query_embeddings=embedding, n_results=self.config["retrieval"]["top_k"]
        )
        return [
            {"text": doc, "source": meta["source"]}
            for doc, meta in zip(result["documents"][0], result["metadatas"][0])
        ]

    def answer(self, question: str) -> dict:
        chunks = self.retrieve(question)
        context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        gen = self.config["generation"]
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=gen["max_new_tokens"],
                do_sample=gen["temperature"] > 0,
                temperature=gen["temperature"] or None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        return {"answer": answer.strip(), "sources": [c["source"] for c in chunks]}
