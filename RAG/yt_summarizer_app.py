import streamlit as st
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ytrag import summarize_youtube_video

def extract_video_id(url):
    # Handles various YouTube URL formats
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"youtu\.be\/([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

st.title("YouTube Video Summarizer (Gemini)")

yt_url = st.text_input("Enter YouTube Video URL:")

if yt_url:
    video_id = extract_video_id(yt_url)
    if not video_id:
        st.error("Invalid YouTube URL.")
    else:
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key or gemini_api_key.strip() == "":
            st.error("Please set the GEMINI_API_KEY in your .env file.")
        else:
            os.environ["GOOGLE_API_KEY"] = gemini_api_key
            try:
                with st.spinner("Summarizing..."):
                    summary = summarize_youtube_video(video_id)
                st.subheader("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Failed to fetch transcript or summarize video. Reason: {e}")
