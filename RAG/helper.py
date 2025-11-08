"""
Minimal RAG helper for PDFs:
- extract_text_from_pdf(pdf_path)
- build_index_from_pdf(pdf_path, index_dir)
- load_index(index_dir)
- answer_query_from_pdf(pdf_path, query, index_dir)
Requires: pypdf, sentence-transformers, faiss (or faiss-cpu), transformers
"""

import os
import pickle
from typing import List, Tuple
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Configuration defaults
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"  # change to a larger/smaller model as desired
CHUNK_SIZE = 500    # approx words per chunk
CHUNK_OVERLAP = 50  # overlap in words
TOP_K = 4

def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
	"""
	Extract per-page text from a PDF.
	Returns list of (page_number, text).
	"""
	reader = PdfReader(pdf_path)
	pages = []
	for i, page in enumerate(reader.pages):
		text = page.extract_text() or ""
		pages.append((i + 1, text))
	return pages

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
	"""
	Split text into overlapping chunks by words.
	"""
	words = text.split()
	if not words:
		return []
	chunks = []
	start = 0
	while start < len(words):
		end = start + chunk_size
		chunk = " ".join(words[start:end])
		chunks.append(chunk)
		if end >= len(words):
			break
		start = end - overlap
	return chunks

def build_index_from_pdf(pdf_path: str, index_dir: str) -> None:
	"""
	Extract text from pdf, chunk, embed, and save FAISS index + metadata to index_dir.
	Produces:
	  - index_dir/index.faiss
	  - index_dir/metadata.pkl
	"""
	os.makedirs(index_dir, exist_ok=True)
	metadata_path = Path(index_dir) / "metadata.pkl"
	index_path = Path(index_dir) / "index.faiss"

	# Extract and chunk
	pages = extract_text_from_pdf(pdf_path)
	chunk_texts = []
	chunk_metas = []  # list of dicts {page, chunk_index}
	for page_no, txt in pages:
		chunks = chunk_text(txt)
		for idx, c in enumerate(chunks):
			chunk_texts.append(c)
			chunk_metas.append({"page": page_no, "chunk_index": idx})

	if not chunk_texts:
		raise ValueError("No text extracted from PDF.")

	# Embed
	emb_model = SentenceTransformer(EMBEDDING_MODEL)
	embs = emb_model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)
	# Normalize embeddings for cosine similarity
	norms = np.linalg.norm(embs, axis=1, keepdims=True)
	norms[norms == 0] = 1.0
	embs = embs / norms

	# Build FAISS index (cosine via inner product on normalized vectors)
	dim = embs.shape[1]
	index = faiss.IndexFlatIP(dim)
	index = faiss.IndexIDMap(index)
	ids = np.arange(len(chunk_texts)).astype("int64")
	index.add_with_ids(embs.astype("float32"), ids)

	# Persist
	faiss.write_index(index, str(index_path))
	with open(metadata_path, "wb") as f:
		pickle.dump({"texts": chunk_texts, "metas": chunk_metas}, f)

def load_index(index_dir: str):
	"""
	Load FAISS index and metadata from index_dir.
	Returns (index, metadata)
	"""
	index_path = Path(index_dir) / "index.faiss"
	metadata_path = Path(index_dir) / "metadata.pkl"
	if not index_path.exists() or not metadata_path.exists():
		raise FileNotFoundError("Index or metadata not found in index_dir. Build index first.")
	index = faiss.read_index(str(index_path))
	with open(metadata_path, "rb") as f:
		meta = pickle.load(f)
	return index, meta

def retrieve_from_index(index, meta, query: str, top_k: int = TOP_K, emb_model=None):
	"""
	Embed query and retrieve top_k chunks. Returns list of (score, text, meta).
	"""
	if emb_model is None:
		emb_model = SentenceTransformer(EMBEDDING_MODEL)
	q_emb = emb_model.encode([query], convert_to_numpy=True)
	# normalize
	q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
	D, I = index.search(q_emb.astype("float32"), top_k)
	scores = D[0].tolist()
	ids = I[0].tolist()
	results = []
	for sc, idx in zip(scores, ids):
		if idx == -1:
			continue
		text = meta["texts"][idx]
		m = meta["metas"][idx]
		results.append((float(sc), text, m))
	return results

def generate_answer(query: str, retrieved: List[Tuple[float, str, dict]], gen_model_name: str = GEN_MODEL, max_length: int = 256) -> str:
	"""
	Use a seq2seq generation model to answer based on retrieved context.
	Concatenates retrieved chunks as context.
	"""
	contexts = []
	for score, text, meta in retrieved:
		contexts.append(f"(page {meta['page']}) {text}")

	prompt = "Use the following extracted document context to answer the question. If the answer is not contained, say you don't know.\n\nContext:\n"
	prompt += "\n---\n".join(contexts)
	prompt += f"\n\nQuestion: {query}\nAnswer:"

	# load generator
	# using transformers pipeline keeps it simple; models load on first call
	# for performance, user can initialize a global pipeline instead.
	tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
	model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
	nlp = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
	out = nlp(prompt, max_length=max_length, do_sample=False)
	return out[0]["generated_text"].strip()

def answer_query_from_pdf(pdf_path: str, query: str, index_dir: str = None, rebuild: bool = False) -> str:
	"""
	Main convenience function: build index if missing (or if rebuild=True), then answer query.
	- pdf_path: path to PDF file
	- query: question string
	- index_dir: directory to store index; if None, uses pdf_path sibling folder with .index
	- rebuild: force rebuild of the index
	Returns generated answer string.
	"""
	if index_dir is None:
		index_dir = str(Path(pdf_path).with_suffix("") ) + ".index"
	if rebuild or not (Path(index_dir) / "index.faiss").exists():
		build_index_from_pdf(pdf_path, index_dir)

	index, meta = load_index(index_dir)
	emb_model = SentenceTransformer(EMBEDDING_MODEL)
	retrieved = retrieve_from_index(index, meta, query, top_k=TOP_K, emb_model=emb_model)
	if not retrieved:
		return "No relevant context retrieved from document."

	answer = generate_answer(query, retrieved, gen_model_name=GEN_MODEL)
	# Optionally include the top retrieved contexts for traceability
	return answer

# Example usage (remove/comment out in production):
ans = answer_query_from_pdf("doc.pdf", "What is the main finding?")
print(ans)
