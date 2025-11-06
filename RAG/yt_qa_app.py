import streamlit as st
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ytrag import get_transcript, split_transcript, build_vector_store, get_retriever, build_prompt
from langchain_google_genai import ChatGoogleGenerativeAI

def extract_video_id(url):
    if not url:
        return None
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})",
        r"^([0-9A-Za-z_-]{11})$"  # Direct video ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

st.title("YouTube Video Question Answering (Gemini)")

yt_url = st.text_input("Enter YouTube Video URL:")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None

if yt_url:
    video_id = extract_video_id(yt_url)
    if not video_id:
        st.error("Invalid YouTube URL. Please provide a valid YouTube URL or video ID.")
    else:
        st.info(f"Extracted video ID: {video_id}")
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key or gemini_api_key.strip() == "":
            st.error("Please set the GEMINI_API_KEY in your .env file.")
        else:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
            if st.session_state.video_id != video_id:
                try:
                    with st.spinner("Processing video transcript..."):
                        st.info("Attempting to fetch video transcript...")
                        transcript = get_transcript(video_id)
                        if not transcript:
                            st.error("No captions available for this video.")
                            st.info("Please try a different video that has captions enabled.")
                            st.session_state.vector_store = None
                            st.session_state.retriever = None
                        else:
                            st.success("Transcript fetched successfully!")
                            chunks = split_transcript(transcript)
                            with st.spinner("Building vector store..."):
                                try:
                                    vector_store = build_vector_store(chunks)
                                    retriever = get_retriever(vector_store)
                                    st.session_state.vector_store = vector_store
                                    st.session_state.retriever = retriever
                                    st.session_state.video_id = video_id
                                    st.success("Ready to answer questions about the video!")
                                except Exception as be:
                                    err_text = str(be)
                                    st.error(f"Error building embeddings/vector store: {err_text}")
                                    # If it's a quota-related error, give specific guidance
                                    if "quota" in err_text.lower() or "billing" in err_text.lower() or "exceed" in err_text.lower():
                                        st.info("Embedding quota appears to be exceeded. Check your Gemini API quota/billing or switch to another embedding provider.")
                                    st.session_state.vector_store = None
                                    st.session_state.retriever = None
                except Exception as e:
                    err_msg = str(e)
                    st.error(f"Error: {err_msg}")
                    # Provide clearer guidance for parse/XML/no-element errors
                    if "transcripts are disabled" in err_msg.lower():
                        st.info("This video does not have any captions available. Please try a different video with captions enabled.")
                    elif "xml" in err_msg.lower() or "no element found" in err_msg.lower() or "parse" in err_msg.lower():
                        st.info("Could not parse the transcript feed. This may indicate the video has no captions or there was a network/formatting issue. Try another video or check network access.")
                    elif "could not translate" in err_msg.lower():
                        st.info("Could not translate the available captions to English. Please try a video with English captions.")
                    elif "embedding quota" in err_msg.lower() or "quota" in err_msg.lower() or "exceeded" in err_msg.lower():
                        st.info("Embedding quota exceeded. Verify your GEMINI_API_KEY, billing, and quota limits.")
                    st.session_state.vector_store = None
                    st.session_state.retriever = None

            if st.session_state.retriever:
                st.markdown("**Ask a question about the video:**")
                with st.form(key="qa_form", clear_on_submit=True):
                    user_q = st.text_input("Your question:", key="qa_input")
                    submit_q = st.form_submit_button("Submit")
                if submit_q and user_q:
                    prompt = build_prompt()
                    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
                    retrieved_docs = st.session_state.retriever.invoke(user_q)
                    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    final_prompt = prompt.invoke({"context": context_text, "question": user_q})
                    with st.spinner("Answering..."):
                        try:
                            answer = llm.invoke(final_prompt)
                            st.markdown("**Answer:**")
                            st.write(answer.content)
                        except Exception as le:
                            le_msg = str(le)
                            st.error(f"Error from LLM: {le_msg}")
                            if "quota" in le_msg.lower() or "exceed" in le_msg.lower():
                                st.info("LLM/embedding quota exceeded. Check Gemini/GCP billing and usage.")
