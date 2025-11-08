import streamlit as st
import requests

st.set_page_config(page_title="Trust Score Generator", page_icon="🤝", layout="centered")

st.title("🤝 Trust Score Generator")
st.markdown("Enter your review text and API Key to generate a Trust Score using the Reviews API.")

st.sidebar.header("🔑 API Settings")
api_key = st.sidebar.text_input("Enter your API Key", type="password")

review_text = st.text_area("🗣️ Enter your review text here:", height=200, placeholder="Type or paste the customer review...")

API_URL = "https://api.reviews.com/v1/trust-score"

if st.button("Generate Trust Score"):
    if not api_key:
        st.error("Please enter your API Key in the sidebar.")
    elif not review_text.strip():
        st.warning("Please enter a review first.")
    else:
        with st.spinner("Analyzing review..."):
            try:
                # Send POST request to API
                payload = {"review": review_text}
                headers = {"Authorization": f"Bearer {api_key}"}

                response = requests.post(API_URL, json=payload, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    trust_score = data.get("trust_score", None)

                    if trust_score is not None:
                        st.success("✅ Trust Score generated successfully!")
                        st.metric(label="Trust Score", value=f"{trust_score:.2f} / 100")

                        # Show progress bar
                        st.progress(int(trust_score))

                        # Interpretation
                        if trust_score >= 80:
                            st.markdown("### 🟢 Excellent Trust")
                        elif trust_score >= 50:
                            st.markdown("### 🟡 Moderate Trust")
                        else:
                            st.markdown("### 🔴 Low Trust")

                        with st.expander("View Raw API Response"):
                            st.json(data)
                    else:
                        st.error("API response did not contain a 'trust_score' field.")
                else:
                    st.error(f"API returned error {response.status_code}: {response.text}")

            except Exception as e:
                st.error(f"Error while connecting to API: {e}")
