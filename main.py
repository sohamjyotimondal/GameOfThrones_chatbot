import streamlit as st
import os
from chatbot import QuoteRAGEngine

st.set_page_config(page_title="Game of Thrones QuoteBot", page_icon="🐉")

# Dark theme CSS
st.markdown(
    """
    <style>
    body, .stApp {
        background-color: #161616 !important;
    }
    </style>
    <h1 style='text-align: center; color: #f7c873;'>❄️ Game of Thrones QuoteBot 🐺</h1>
    <p style='text-align: center; color: #cccccc; font-size:1.2em;'>
       Valar Morghulis
    </p>
    <hr style='border: 1px solid #2a2a2a;'>
    """,
    unsafe_allow_html=True,
)


if "user_id" not in st.session_state:
    st.session_state.user_id = "toxicplutonite"


@st.cache_resource(
    show_spinner="Waking up the maesters and loading your chat history..."
)
def load_engine(user_id):
    """Load the chat engine with user-specific chat history."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Please set the GROQ_API_KEY environment variable.")
        st.stop()
    return QuoteRAGEngine(groq_api_key=groq_api_key, user_id=user_id)


# Load engine with user ID
engine = load_engine(st.session_state.user_id)

# Initialize Streamlit session state for UI
if "messages" not in st.session_state:
    st.session_state.messages = []

    try:
        existing_history = engine.get_chat_history()
        if existing_history:
            for msg in existing_history:
                role = "user" if msg.role.value == "user" else "assistant"
                st.session_state.messages.append({"role": role, "content": msg.content})
    except Exception as e:
        print(f"Could not load existing history: {e}")

# Sidebar
with st.sidebar:
    st.title("Chat Management")

    st.info(f"Current User: {st.session_state.user_id}")

    st.subheader("Chat Controls")

    if st.button("🗑️ Clear Chat History", type="secondary"):

        st.session_state.messages = []
        engine.clear_chat_history()
        st.success("Chat history cleared!")
        st.rerun()

    if st.button("💾 Save & Sync History"):
        engine.load_chat_history_from_streamlit(st.session_state.messages)
        st.success("Chat history synced!")

    st.subheader("Chat Stats")
    total_messages = len(st.session_state.messages)
    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    bot_messages = len(
        [m for m in st.session_state.messages if m["role"] == "assistant"]
    )

    st.metric("Total Messages", total_messages)
    st.metric("Your Messages", user_messages)
    st.metric("Bot Replies", bot_messages)


def render_message(who, msg):
    """Render a chat message with styling."""
    align = "right" if who == "user" else "left"
    avatar = "" if who == "user" else "🐺"
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


# Display chat messages
with st.container():
    if not st.session_state.messages:  # default message without any history
        render_message(
            "assistant",
            "Winter is coming... but I'm here to chat! Ask me anything, and I'll respond with the wisdom of Westeros.",
        )
    else:
        for message in st.session_state.messages:
            render_message(message["role"], message["content"])


user_input = st.chat_input("Type your message and press Enter...")

if user_input:
    # Add message under role user
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Get bot response
    with st.spinner("Summoning a quote from Westeros..."):
        try:
            reply = engine.get_reply(user_input)
            reply = reply.strip('"')  # Clean up quotes
        except Exception as e:
            reply = "The old gods seem silent... Perhaps try rephrasing your message?"
            print(f"Error: {e}")

    # Add response to role assistant (bot)
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Automatically save the conversation
    try:
        engine.save_chat_history()
    except Exception as e:
        print(f"Could not save chat history: {e}")

    st.rerun()

# Footer information
# st.markdown(
#     """
#     <hr style='border: 1px solid #2a2a2a; margin-top: 2rem;'>
#     <p style='text-align: center; color: #666; font-size: 0.8em;'>
#        The king in the North...<br>
#     </p>
#     """,
#     unsafe_allow_html=True,
# )
