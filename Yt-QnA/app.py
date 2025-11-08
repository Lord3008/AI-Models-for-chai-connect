import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import re
import math
import time
from typing import List, Tuple

st.set_page_config(page_title="YouTube Video Q&A (RAG + Gemini)", layout="wide")


def extract_video_id(url: str) -> str:
    """
    Extract YouTube video id from typical URL forms.
    """
    # common patterns
    patterns = [
        r"v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"youtube\.com/embed/([A-Za-z0-9_-]{11})",
        r"youtube\.com/watch\?.*&v=([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    # If user typed a raw id
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    raise ValueError("Could not extract a YouTube video id from the provided URL.")

def fetch_transcript(video_id: str) -> str:
    """
    Fetch transcript with retries and better error messages.
    """
    import time
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

    for attempt in range(3):
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            lines = [f"[{int(seg.get('start', 0))}s] {seg.get('text','').strip()}"
                     for seg in transcript_list if seg.get('text')]
            return "\n".join(lines)
        except TranscriptsDisabled:
            raise RuntimeError("❌ Transcripts are disabled for this video.")
        except NoTranscriptFound:
            raise RuntimeError("❌ No transcript found for this video.")
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
                continue
            raise RuntimeError(f"❌ Failed to fetch transcript after retries. "
                               f"This may happen if the video is private, region-locked, "
                               f"or has no captions. Details: {e}")



def chunk_text(text: str, chunk_size_words: int = 120, overlap_words: int = 20) -> List[str]:
    """
    Naive chunking by words with overlap, returns list of chunks (strings).
    Keeps segment boundaries intact as far as possible by splitting on newlines.
    """
    tokens = text.split()
    chunks = []
    i = 0
    n = len(tokens)
    while i < n:
        j = min(i + chunk_size_words, n)
        chunk = " ".join(tokens[i:j])
        chunks.append(chunk)
        i = j - overlap_words 
        if i < 0:
            i = 0
    return chunks

def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """
    Return a numpy array of shape (len(texts), dim)
    """
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype("float32")

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build an index for L2 similarity (we assume vectors are normalized so L2 works similarly to cosine).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def retrieve_top_k(index: faiss.Index, query_vector: np.ndarray, texts: List[str], k: int = 4) -> List[Tuple[str, float]]:
    """
    Returns list of (text, score) for top k results.
    """
    if index.ntotal == 0:
        return []
    # query_vector shape: (dim,)
    q = np.copy(query_vector).astype("float32").reshape(1, -1)
    scores, idxs = index.search(q, k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        results.append((texts[int(idx)], float(score)))
    return results

def call_gemini_generate(api_key: str, model_name: str, prompt: str, temperature: float = 0.0, max_tokens: int = 512):
    """
    Generic Gemini/PaLM REST call wrapper.
    IMPORTANT: model_name should be the model id you have access to,
    e.g. 'text-bison-001' or 'gemini-1.5' or similar. If your account uses another path/version,
    update the URL accordingly.

    The function tries the common v1beta2 generateText style. If your model endpoint is different,
    swap the request body/URL as required by your account.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Default endpoint used here: v1beta2 generate (if your account uses v1 or v1beta3, change the base)
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/gemini-2.0-flash:generateText"

    body = {
        "prompt": {
            "text": prompt
        },
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
    }

    resp = requests.post(url, headers=headers, json=body, timeout=30)
    try:
        j = resp.json()
    except Exception:
        resp.raise_for_status()
    if resp.status_code not in (200, 201):
        # surface the returned JSON for debugging
        return {"error": True, "status_code": resp.status_code, "response": j}
    # Depending on model version, the field may be 'candidates' or 'output' etc.
    # We'll try to extract the common fields gracefully.
    # For many PaLM-like responses:
    #   j.get("candidates", [{}])[0].get("output", "...") or j.get("output", "")
    output = None
    if isinstance(j, dict):
        # try a few known shapes
        if "candidates" in j and isinstance(j["candidates"], list) and len(j["candidates"])>0:
            output = j["candidates"][0].get("content") or j["candidates"][0].get("output") or j["candidates"][0].get("text")
            if isinstance(output, list):
                # sometimes content is list of {type, text}
                out_text = ""
                for c in output:
                    if isinstance(c, dict) and "text" in c:
                        out_text += c["text"]
                output = out_text
        if not output and "output" in j:
            if isinstance(j["output"], str):
                output = j["output"]
            elif isinstance(j["output"], list):
                # join
                output = " ".join([chunk.get("content", "") if isinstance(chunk, dict) else str(chunk) for chunk in j["output"]])
        if not output:
            # fallback: try 'text' fields deep search
            if "text" in j:
                output = j.get("text")
    return {"error": False, "raw": j, "text": output or ""}


# -------------------------
# Streamlit UI
# -------------------------
st.title("YouTube Video Q&A — RAG with local embeddings + Gemini")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Gemini API Key (Bearer)", type="password", placeholder="Paste your Gemini API key here")
    model_name = st.text_input("Gemini model name", value="text-bison-001",
                               help="Set the model id you have access to (e.g. text-bison-001, gemini-1.5).")
    chunk_size = st.number_input("Chunk size (words)", min_value=50, max_value=600, value=120)
    overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=200, value=20)
    top_k = st.number_input("Top-k retrieved chunks", min_value=1, max_value=10, value=4)
    temperature = st.slider("Gemini temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

st.markdown("Paste a YouTube URL or a video id, click **Fetch transcript**. The app will create an embedding index locally.")
col1, col2 = st.columns([3, 1])
with col1:
    video_input = st.text_input("YouTube URL or Video ID")
with col2:
    if st.button("Fetch transcript"):
        if not video_input:
            st.error("Please enter a YouTube URL or video id.")
        else:
            try:
                video_id = extract_video_id(video_input)
                st.session_state['video_id'] = video_id
                with st.spinner("Fetching transcript..."):
                    transcript_text = fetch_transcript(video_id)
                st.session_state['transcript_text'] = transcript_text
                st.success("Transcript fetched and stored in session.")
            except Exception as e:
                st.exception(e)

if 'transcript_text' in st.session_state:
    st.subheader("Transcript (first 2000 chars)")
    st.text_area("Transcript preview", value=st.session_state['transcript_text'][:2000], height=220)
    if st.button("Build RAG index from transcript"):
        with st.spinner("Chunking and building embeddings (this may take a few seconds)..."):
            transcript_text = st.session_state['transcript_text']
            chunks = chunk_text(transcript_text, chunk_size_words=int(chunk_size), overlap_words=int(overlap))
            st.session_state['chunks'] = chunks
            # load sentence-transformers model
            model = SentenceTransformer('all-MiniLM-L6-v2')
            st.session_state['embed_model_name'] = 'all-MiniLM-L6-v2'
            embs = embed_texts(model, chunks)  # normalized floats
            st.session_state['embeddings'] = embs
            # build faiss
            try:
                index = build_faiss_index(embs)
                st.session_state['faiss_index'] = index
            except Exception as e:
                st.warning("FAISS index build failed. You may be on a platform without faiss support. Falling back to numpy brute-force.")
                st.session_state['faiss_index'] = None
            st.success(f"Built index with {len(chunks)} chunks.")

st.markdown("---")
st.subheader("Ask a question about the video")
question = st.text_input("Your question")

if st.button("Get Answer"):

    if 'faiss_index' not in st.session_state or 'chunks' not in st.session_state or 'embeddings' not in st.session_state:
        st.error("No index found. Please fetch a transcript and build the RAG index first.")
    elif not api_key:
        st.error("Please provide your Gemini API Key in the sidebar.")
    elif not question:
        st.error("Please type a question.")
    else:
        # embed the question with same model
        with st.spinner("Retrieving relevant transcript chunks..."):
            model = SentenceTransformer(st.session_state.get('embed_model_name', 'all-MiniLM-L6-v2'))
            q_emb = model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            faiss_index = st.session_state.get('faiss_index', None)
            if faiss_index is None:
                # brute-force cosine via numpy
                embs = st.session_state['embeddings']  # already normalized
                sims = (embs @ q_emb.T).squeeze()
                idxs = np.argsort(-sims)[:int(top_k)]
                retrieved = [(st.session_state['chunks'][int(i)], float(sims[int(i)])) for i in idxs]
            else:
                retrieved = retrieve_top_k(faiss_index, q_emb.squeeze(), st.session_state['chunks'], k=int(top_k))

        # prepare context
        context_parts = []
        for i, (txt, score) in enumerate(retrieved):
            context_parts.append(f"--- chunk {i+1} (score={score:.4f}) ---\n{txt}\n")
        context = "\n".join(context_parts)
        # craft prompt for Gemini
        prompt = f"""
You are a helpful assistant. Use ONLY the provided transcript/context to answer the user's question. If the answer isn't present in the context, say you don't know and avoid guessing.

Context:
{context}

User question:
{question}

Answer concisely, include timestamps or quoted lines from the context if useful. If the context is insufficient, say \"I don't know based on the transcript\".
"""
        st.markdown("**Retrieved context (top chunks)**")
        for idx, (txt, score) in enumerate(retrieved):
            st.markdown(f"**Chunk {idx+1}** (score={score:.4f})")
            st.write(txt[:800] + ("..." if len(txt) > 800 else ""))

        st.markdown("**Calling Gemini to generate the final answer...**")
        with st.spinner("Calling Gemini..."):
            resp = call_gemini_generate(api_key=api_key, model_name=model_name, prompt=prompt, temperature=float(temperature), max_tokens=512)
        if resp.get("error"):
            st.error(f"Gemini API returned error (HTTP {resp.get('status_code')}):")
            st.json(resp.get("response"))
            st.write("If you see a 404 or model-not-found error, check that `model_name` is correct for your account and that your API key is valid. Try common model names like `text-bison-001`, `gemini-1.5`, or check your Google Cloud console for the exact model id.")
        else:
            text = resp.get("text")
            if not text:
                st.warning("No text found in Gemini response. Here is the raw response JSON for debugging:")
                st.json(resp.get("raw"))
            else:
                st.success("Answer generated by Gemini:")
                st.markdown(text)

        # offer raw response for debugging
        if st.checkbox("Show raw Gemini JSON response"):
            st.json(resp.get("raw"))

st.markdown("---")
st.caption("Built with youtube-transcript-api + sentence-transformers + FAISS (local embeddings) + Gemini for generation. "
           "If your environment can't install faiss, you can modify the retrieval to use numpy dot products instead (I included a fallback).")
