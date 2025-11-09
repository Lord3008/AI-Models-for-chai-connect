# YouTube RAG Apps — Architecture & Usage

This document explains the architecture and runtime flow for the three RAG (Retrieval-Augmented Generation) components:
- yt_qa_app.py — Streamlit app for question-answering about a YouTube video using transcript context + Gemini LLM.
- ytrag.py — Core utilities: transcript fetching, cleaning, chunking, embeddings, vector store, retriever, and prompt builder.
- yt_summarizer_app.py — Streamlit app that summarizes a YouTube video using the same pipeline and LLM.

High-level architecture
1. Input (UI)
   - User provides a YouTube URL or video ID via Streamlit UI (yt_qa_app.py or yt_summarizer_app.py).
   - Environment variable GEMINI_API_KEY (or GOOGLE_API_KEY) must be set.

2. Transcript extraction (ytrag.get_transcript)
   - Uses yt_dlp to extract video metadata and find subtitle/caption URLs.
   - Prioritizes manual English subtitles, then automatic captions.
   - Requests the subtitle feed and converts VTT/SRT to plain text (timestamps removed).
   - Returns None if no usable transcript is available or if network/extraction fails.

3. Text chunking (ytrag.split_transcript)
   - Uses langchain RecursiveCharacterTextSplitter to produce context chunks (configurable chunk_size and overlap).
   - Chunks are small documents used for embedding and retrieval.

4. Embeddings & Vector store (ytrag.build_vector_store)
   - Embeddings created via GoogleGenerativeAIEmbeddings (Gemini embedding model in code).
   - Creates a FAISS vector store from chunked documents for fast similarity search.
   - Error handling detects quota/billing messages and surfaces friendly guidance.

5. Retriever & Prompt (ytrag.get_retriever, ytrag.build_prompt)
   - Retriever exposes a similarity search API (k nearest neighbors).
   - PromptTemplate enforces "Answer ONLY from provided transcript context" and supplies context + question to LLM.

6. LLM Calls (ChatGoogleGenerativeAI)
   - The app constructs final_prompt with retrieved context and question, and calls Gemini (or configured LLM).
   - yt_qa_app passes user question and prints the LLM's answer.
   - yt_summarizer_app sends a summarization prompt and displays the result.

Runtime flow in Streamlit apps
- yt_qa_app.py:
  - Extract video_id, verify API key.
  - If vector store not cached in session_state or video changed:
    - Fetch transcript → split → build vector store → create retriever → cache in session_state.
  - On user question:
    - Retrieve top-k documents → build prompt → call LLM → show answer.
  - Handles common failure modes (no captions, parse errors, embedding/LLM quota) and surfaces guidance.

- yt_summarizer_app.py:
  - Similar transcript → chunk → embed → retrieve pipeline.
  - Calls summarize_youtube_video which runs retrieval + LLM summarization.

Key implementation details & decisions
- Transcript cleaning strips timecodes and VTT/SRT artifacts to produce continuous text.
- Chunking with overlap preserves context across chunk boundaries.
- FAISS used for local, memory-efficient nearest-neighbor retrieval.
- GoogleGenerativeAIEmbeddings + ChatGoogleGenerativeAI (Gemini) are used; code detects and surfaces quota or billing errors.
- Session-level caching in Streamlit avoids re-building vector stores for repeated requests on same video.

Error handling & operator guidance
- Check GEMINI_API_KEY / GOOGLE_API_KEY in .env for LLM and embedding calls.
- If embedding quota errors appear, switch embedding provider or increase project quota.
- If transcript is missing, try a video with captions enabled or enable auto captions on the source.

Requirements
- Python 3.8+
- Packages: streamlit, yt_dlp, requests, langchain, langchain-community, langchain-core, google generative client wrappers used in code, faiss (or langchain FAISS wrapper), tensorflow/torch not required for these scripts.
- .env with GEMINI_API_KEY or GOOGLE_API_KEY (and billing enabled for embedding/LLM usage).

Deployment & scaling notes
- For production, move vector store persistence to disk or a shared datastore (e.g., Pinecone, Milvus) instead of ephemeral in-memory FAISS.
- Use per-video vector store caching and expiration policy.
- Rate-limit LLM/embedding calls and add retry/backoff for transient errors.
- Secure API keys in secret manager; avoid embedding keys in code or client-side UI.
- Consider cost controls: usage quotas, request batching, cheaper embedding models if available.

Extensibility
- Swap embedding/LLM provider by implementing a thin adapter in ytrag.build_vector_store and calling a different Chat wrapper.
- Add language detection & translation when captions are non-English.
- Enhance prompt engineering to include chain-of-thought, citation of transcript chunk indices, or provenance metadata for answers.

Contact
- See repository for sample .env, example usage, and further developer notes.
