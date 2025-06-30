import json
import os
from typing import List, Dict, Optional
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.llms import ChatMessage, MessageRole

QUOTES_PATH = "datas.json"
INDEX_DIR = "quotes_index"
CHAT_STORE_PATH = "chat_history"


def load_quotes(path: str = QUOTES_PATH) -> List[Dict]:
    with open(path, "r") as f:
        quotes_json = json.load(f)
        return quotes_json["quotes"]


def quotes_to_documents(quotes: List[Dict]) -> List[Document]:
    return [
        Document(text=q["sentence"], extra_info={"character": q["character"]})
        for q in quotes
    ]


def has_persistent_index(index_dir: str) -> bool:
    required_files = ["docstore.json", "default__vector_store.json", "index_store.json"]
    return os.path.exists(index_dir) and all(
        os.path.isfile(os.path.join(index_dir, f)) for f in required_files
    )


class QuoteRAGEngine:
    def __init__(
        self,
        groq_api_key: str,
        quotes_path: str = QUOTES_PATH,
        index_dir: str = INDEX_DIR,
        chat_store_path: str = CHAT_STORE_PATH,
        user_id: Optional[str] = "default_user",
    ):
        self.groq_api_key = groq_api_key
        self.quotes_path = quotes_path
        self.index_dir = index_dir
        self.chat_store_path = chat_store_path
        self.user_id = user_id

        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        # Load quotes and build index
        self.quotes = load_quotes(self.quotes_path)
        self.index = self.build_or_load_index()

        # Initialize chat components
        self.chat_store = self.setup_chat_store()
        self.memory = self.setup_memory()
        self.chat_engine = self.build_chat_engine()

    def build_or_load_index(self):
        """Build or load the vector index for quotes."""
        if not has_persistent_index(self.index_dir):
            if not os.path.exists(self.index_dir):
                os.makedirs(self.index_dir, exist_ok=True)
            docs = quotes_to_documents(self.quotes)
            index = VectorStoreIndex.from_documents(
                docs, embed_model=self.embed_model, show_progress=True
            )
            index.storage_context.persist(self.index_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
            index = load_index_from_storage(
                storage_context, embed_model=self.embed_model
            )
        return index

    def setup_chat_store(self):
        """Setup chat store for persistent chat history."""
        # Create chat store directory if it doesn't exist
        if not os.path.exists(self.chat_store_path):
            os.makedirs(self.chat_store_path, exist_ok=True)

        chat_store_file = os.path.join(
            self.chat_store_path, f"{self.user_id}_chat.json"
        )

        # Load existing chat store or create new one
        if os.path.exists(chat_store_file):
            chat_store = SimpleChatStore.from_persist_path(chat_store_file)
        else:
            chat_store = SimpleChatStore()

        return chat_store

    def setup_memory(self):
        """Setup memory buffer with chat store."""
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,  # Adjust based on your needs
            chat_store=self.chat_store,
            chat_store_key=self.user_id,
        )
        return memory

    def build_chat_engine(self):
        """Build the chat engine with memory."""
        llm = Groq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            api_key=self.groq_api_key,
            temperature=0.7,
        )


        chat_engine = self.index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=self.memory,
            llm=llm,
            context_prompt=(
                "You are a witty, in-character Game of Thrones chatbot. "
                "For every user message, use the provided quotes as context. "
                "Reply with the most fitting quote (or adapt a quote) and sound like a real Thrones character. "
                "If you can't find a fitting quote, say something mysterious."
                "For example if the user says 'It is so cold', you might reply with 'Winter is coming.' "
                "If the user says 'There are so many ants', you might reply with 'Burn them all.' "
                "If the user says 'I lost in chess', you can reply with 'When you play the game of chess, you win or you die. There is no middle ground.' "
                "As you can see you must keep closely to the context but you may change it a bit. "
                """IMPORTANT: Always see it is a good reply to the user and not just a quote.
                example: If user says "How to save a life" you should not reply with 'Don't let me die, Jaime. Please don't let me die.'
                because even if it is a quote it does not make sense in this context see that the conversation is in context. You can encourage killing jokingly as you are a dank bot and no one will take it seriously
                example: If user says 'kill yourself' The dont say 'You're shit at dying, you know that?' as that is again a bit out of context rather say 'I'm shit at dying, you know that?' as the user is
                saying you to kill yourself so it makes more sense in the context. Make it seem like talking to a human.
                Follow the quotes but they are not absolute you can change them accordingly\n
                Given below is the chat history"""
                "{context_str}"
                "\nInstruction: The previous chat history is given but always focus on the user message and the quotes. "
                "Just keep the chat history as a guide in case you need any aditional context otherwise discard it."
                "**IMPORTANT:Always give priority to the current question and always provide  short dank answers**"
            ),
            condense_prompt=(
                "Given the conversation history and a follow-up question, "
                "Keep the question as it is and if and only if the previous chat history is needed condense"
                "the context of the conversation with the new question. The current question always takes priority "
                " Make it dank and witty always"
            ),
            verbose=False,
        )
        return chat_engine

    def get_reply(self, user_message: str) -> str:
        """Get a reply from the chat engine with conversation history."""
        try:
            response = self.chat_engine.chat(user_message)
            self.save_chat_history()

            return str(response)
        except Exception as e:
            print(f"Error generating response: {e}")
            return "The realm seems troubled... try asking again, my lord."

    def save_chat_history(self):
        """Save chat history to disk."""
        chat_store_file = os.path.join(
            self.chat_store_path, f"{self.user_id}_chat.json"
        )
        self.chat_store.persist(chat_store_file)

    def clear_chat_history(self):
        """Clear the chat history for the current user."""
        self.chat_engine.reset()
        self.save_chat_history()

    def get_chat_history(self) -> List[ChatMessage]:
        """Get the current chat history."""
        return self.memory.get_all()

    def set_chat_history(self, chat_history: List[ChatMessage]):
        """Set the chat history manually."""
        self.memory.set(chat_history)
        self.save_chat_history()

    def load_chat_history_from_streamlit(self, streamlit_history: List[Dict]):
        """Convert Streamlit chat history to LlamaIndex format and load it."""
        chat_messages = []
        for msg in streamlit_history:
            role = MessageRole.USER if msg["role"] == "user" else MessageRole.ASSISTANT
            chat_messages.append(ChatMessage(role=role, content=msg["content"]))

        self.set_chat_history(chat_messages)


if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set the GROQ_API_KEY environment variable.")
    else:
        # Example usage with different user IDs
        user_id = (
            input("Enter your user ID (or press enter for default): ").strip()
            or "toxicplutonite"
        )

        engine = QuoteRAGEngine(groq_api_key=api_key, user_id=user_id)

        print(f"Game of Thrones Chat Bot (User: {user_id})")
        print(
            "Type 'clear' to clear chat history, 'history' to see chat history, or 'quit' to exit"
        )
        print("-" * 50)

        while True:
            user_query = input("You: ")

            if user_query.lower() == "quit":
                break
            elif user_query.lower() == "clear":
                engine.clear_chat_history()
                print("Chat history cleared!")
                continue
            elif user_query.lower() == "history":
                history = engine.get_chat_history()
                print("\nChat History:")
                for msg in history:
                    role = "You" if msg.role == MessageRole.USER else "Bot"
                    print(f"{role}: {msg.content}")
                print("-" * 30)
                continue

            reply = engine.get_reply(user_query)
            print(f"Bot: {reply}")
