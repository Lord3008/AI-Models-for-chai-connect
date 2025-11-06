import yt_dlp
import requests
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

import os

def _choose_sub_url(tracks):
    if not tracks:
        return None, None
    # Prefer English variants
    for pref in ("en", "en-US", "en-GB"):
        if pref in tracks and tracks[pref]:
            return tracks[pref][0].get("url"), pref
    # Fallback: first available
    for lang, formats in tracks.items():
        if formats:
            return formats[0].get("url"), lang
    return None, None

def _strip_vtt_or_srt(text):
    lines = text.splitlines()
    out = []
    timestamp_re = re.compile(r"^\s*\d{1,3}\s*$|^\d{2}:\d{2}:\d{2}[.,]\d+|-->|^WEBVTT", re.IGNORECASE)
    for line in lines:
        if timestamp_re.search(line):
            continue
        line = line.strip()
        if not line:
            continue
        out.append(line)
    return " ".join(out)

def get_transcript(video_id):
    if not video_id:
        raise ValueError("Invalid video ID")

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True}

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        # Could be network or extractor issue — treat as no captions for friendly UI flow
        return None

    # Try manual subtitles first, then automatic captions
    sub_url, lang = _choose_sub_url(info.get("subtitles") or {})
    if not sub_url:
        sub_url, lang = _choose_sub_url(info.get("automatic_captions") or {})

    if not sub_url:
        return None

    try:
        resp = requests.get(sub_url, timeout=10)
        if not resp.ok or not resp.text:
            return None
        text = resp.text
        # Convert VTT/SRT to plain text
        plain = _strip_vtt_or_srt(text)
        return plain if plain else None
    except Exception:
        return None

def split_transcript(transcript, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([transcript])

def build_vector_store(chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        msg = str(e).lower()
        # Detect common quota/limit messages and return a clearer error
        if "quota" in msg or "exceeded" in msg or "embed_content_free_tier_requests" in msg:
            raise Exception(
                "Embedding quota exceeded for the Gemini API. "
                "Check your Google Cloud / Gemini billing and quota, or switch to an alternative embedding provider."
            )
        # Re-raise other errors for debugging/visibility
        raise

def get_retriever(vector_store, k=4):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})

def build_prompt():
    return PromptTemplate(
        template="""
          You are a helpful assistant.
          Answer ONLY from the provided transcript context.
          If the context is insufficient, just say you don't know.

          {context}
          Question: {question}
        """,
        input_variables=['context', 'question']
    )

def summarize_youtube_video(video_id, question="Summarize the video"):
    transcript = get_transcript(video_id)
    if not transcript:
        return "No captions available for this video."
    chunks = split_transcript(transcript)
    try:
        vector_store = build_vector_store(chunks)
    except Exception as e:
        # Propagate a friendly message so callers (UI) can display guidance
        raise Exception(str(e))
    retriever = get_retriever(vector_store)
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = build_prompt()
    final_prompt = prompt.invoke({"context": context_text, "question": question})
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    answer = llm.invoke(final_prompt)
    return answer.content
