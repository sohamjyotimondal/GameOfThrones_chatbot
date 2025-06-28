import streamlit as st
import os
from chatbot import QuoteRAGEngine

st.set_page_config(page_title="Game of Thrones QuoteBot", page_icon="🐉")
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #161616 !important;
    }
    </style>
    <h1 style='text-align: center; color: #f7c873;'>❄️ Game of Thrones QuoteBot 🐺</h1>
    <p style='text-align: center; color: #cccccc; font-size:1.2em;'>
        Get Game Of Thrones quotes to impress the huzz
    </p>
    <hr style='border: 1px solid #2a2a2a;'>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Waking up the maesters...")
def load_engine():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Please set the GROQ_API_KEY environment variable.")
        st.stop()
    return QuoteRAGEngine(groq_api_key=groq_api_key)


engine = load_engine()

if "history" not in st.session_state:
    st.session_state.history = []


def render_message(who, msg):
    align = "right" if who == "user" else "left"
    avatar = "" if who == "user" else "🐺"
    # Choose dark-friendly colors:
    # User: dark blue-gray bubble, white text
    # Bot: deep gold bubble, almost black text
    bgcolor = "#232946" if who == "user" else "#f7c873"
    fontcolor = "#ffffff" if who == "user" else "#2b1907"
    st.markdown(
        f"""
        <div style="background-color:{bgcolor};
                    color:{fontcolor};
                    padding:10px 18px;
                    border-radius:14px;
                    margin:10px 0;
                    max-width:80%;
                    float:{align};
                    clear:both;
                    box-shadow:0 2px 6px rgba(0,0,0,0.18);">
            <span style="font-size:1.3em">{avatar}</span>
            <span style="font-size:1.13em">{msg}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.container():
    for who, msg in st.session_state.history:
        render_message(who, msg)

user_input = st.chat_input("Type your message and press Enter...")

if user_input:
    st.session_state.history.append(("user", user_input))
    with st.spinner("Summoning a quote from Westeros..."):
        reply = engine.get_reply(user_input)
        reply = reply.strip('"')  # Clean up quotes
    st.session_state.history.append(("bot", reply))
    st.rerun()
