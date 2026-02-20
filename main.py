import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from PIL import Image
import requests
from io import BytesIO
import urllib.parse
import time

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="TRIP NEXUS PRO",
    page_icon="✈️",
    layout="wide"
)

# ==================================================
# FULL CSS – COLOR FIX GUARANTEED
# ==================================================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #081c3a, #001a33);
    color: white;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001a33, #00264d);
    color: white;
}
[data-testid="stChatMessage"] {
    background-color: #002b5c !important;
    padding: 14px;
    border-radius: 15px;
    margin-bottom: 12px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.5);
}
[data-testid="stChatMessage"]
[data-testid="stMarkdownContainer"] * {
    color: #ffffff !important;
    opacity: 1 !important;
}
div[data-testid="stChatInput"] textarea {
    background-color: #003366 !important;
    color: white !important;
    border-radius: 12px !important;
}
.stButton>button {
    background-color: #004080;
    color: white;
    border-radius: 10px;
}
.title-style {
    font-size: 50px;
    font-weight: bold;
    text-align: center;
    color: white;
    text-shadow: 3px 3px 15px rgba(0,170,255,0.9);
}
.sub-style {
    text-align: center;
    color: #cce6ff;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# TITLE
# ==================================================
st.markdown("""
<div style="display:flex; justify-content:center; gap:15px;">
    <img src="https://cdn-icons-png.flaticon.com/512/201/201623.png" width="60">
    <div class="title-style">TRIP NEXUS PRO</div>
</div>
<div class="sub-style">
The brain behind better trips ✈️
</div>
""", unsafe_allow_html=True)

# ==================================================
# SESSION STATE
# ==================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==================================================
# IMAGE GENERATOR (AUTO PLACE IMAGE)
# ==================================================
def generate_place_image(query):
    try:
        encoded = urllib.parse.quote(query)
        url = f"https://image.pollinations.ai/prompt/{encoded}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
            return Image.open(BytesIO(response.content))
        else:
            return None
    except:
        return None

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("### 📜 Travel Chat History")
    for msg in st.session_state.messages:
        st.markdown(f"• {msg['content'][:40]}")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 🌟 Quick Suggestions")

    if st.button("🏖 Beach Resorts in Goa"):
        st.session_state.quick_question = "Best beach resorts in Goa"

    if st.button("🏔 Hill Stations in India"):
        st.session_state.quick_question = "Top hill stations in India"

    if st.button("🌆 Luxury Resorts in Dubai"):
        st.session_state.quick_question = "Luxury resorts in Dubai"

# ==================================================
# FAST LLM CHAIN
# ==================================================
@st.cache_resource
def get_chain():
    model = OllamaLLM(
        model="gemma3:latest",
        temperature=0.6,
        num_predict=300
    )

    template = """
You are TRIP NEXUS, a professional AI travel consultant.

Question:
{question}

Respond in 5 short bullet points.
Be concise and professional.
"""

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | model

chain = get_chain()

# ==================================================
# RETRIEVER
# ==================================================
@st.cache_data(show_spinner=False)
def get_records(query):
    return retriever.invoke(query)

# ==================================================
# SHOW OLD CHAT
# ==================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==================================================
# USER INPUT
# ==================================================
user_input = st.chat_input("🔎 Search destinations, resorts, packages...")

if "quick_question" in st.session_state:
    user_input = st.session_state.quick_question
    del st.session_state.quick_question

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("✈️ Exploring the world for you..."):

            # Retriever Logic
            if len(user_input.split()) >= 8:
                records = get_records(user_input)
            else:
                records = ""

            response = chain.invoke({
                "question": user_input
            })

            # 🔥 Generate Image Based on Query
            image = generate_place_image(user_input)

            if image:
                st.image(image, use_container_width=True)

            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})