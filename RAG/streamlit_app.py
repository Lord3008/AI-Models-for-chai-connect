import streamlit as st
from decorator import process_file_with_gemini, generate_questions
import re

st.set_page_config(page_title="Gemini Formatter", layout="centered")

st.title("📄 Gemini File Formatter")
st.write(
    "Upload a PDF or TXT file and get a fun, engaging summary with emojis and gaps! "
    "Powered by Gemini API."
)

uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    # Save uploaded file to a temporary location
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    try:
        with st.spinner("Processing..."):
            result = process_file_with_gemini(temp_path)
            mcqs, brainstorming = generate_questions(temp_path)
        # Show formatted text and questions side by side
        col1, col2 = st.columns(2)
        with col1:
            st.success("Here is your formatted text:")
            st.markdown(result)
        with col2:
            st.subheader("📝 MCQs")
            mcq_counter = 0
            for q in mcqs:
                q_clean = re.sub(r"[*#>`]", "", q).strip()
                if not q_clean:
                    continue  # Skip empty questions
                lines = [line.strip() for line in q_clean.split('\n') if line.strip()]
                if len(lines) > 1:
                    st.markdown(f"{mcq_counter}. {lines[0]}")
                    options = []
                    for opt in lines[1:]:
                        opt = re.sub(r"[*#>`]", "", opt)
                        opt = re.sub(r"\(Correct\)|✔️", "", opt, flags=re.IGNORECASE).strip()
                        if opt:
                            options.append(opt)
                    if options:
                        st.markdown('\n'.join([f"- {opt}" for opt in options]))
                else:
                    st.markdown(f"{mcq_counter}. {q_clean}")
                mcq_counter += 1
            st.subheader("💡 Brainstorming Questions")
            brainstorm_counter = 1
            for q in brainstorming:
                q_clean = re.sub(r"[*#>`]", "", q).strip()
                if not q_clean:
                    continue  # Skip empty questions
                st.markdown(f"{brainstorm_counter}. {q_clean}")
                brainstorm_counter += 1
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
else:
    st.info("Please upload a file.")
