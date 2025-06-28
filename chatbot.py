import json
import os
from typing import List, Dict
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.query_engine import RetrieverQueryEngine

QUOTES_PATH = "datas.json"
INDEX_DIR = "quotes_index"


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
    ):
        self.groq_api_key = groq_api_key
        self.quotes_path = quotes_path
        self.index_dir = index_dir
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        self.quotes = load_quotes(self.quotes_path)
        self.index = self.build_or_load_index()
        self.query_engine = self.build_query_engine()

    def build_or_load_index(self):
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

    def build_query_engine(self):
        llm = Groq(
            model="llama3-70b-8192",
            api_key=self.groq_api_key,
            temperature=0.7,
            system_prompt=(
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
                 example:  If user says 'kill yourself' The dont say 'You're shit at dying, you know that?' as that is again a bit out of context rather say 'I'm shit at dying, you know that?' as the user is 
                 saying you to kill yourself so it makes more sense in the context. Make it seem like talking to a human.
                 Follow the quotes but they are not absolute you can change them accordingly"""
            ),
        )
        retriever = self.index.as_retriever(similarity_top_k=5)
        response_synthesizer = get_response_synthesizer(llm=llm)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        return query_engine

    def get_reply(self, user_message: str):
        response = self.query_engine.query(user_message)
        return str(response)


if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Please set the GROQ_API_KEY environment variable.")
    else:
        engine = QuoteRAGEngine(groq_api_key=api_key)
        while True:
            # Demo query
            user_query = input()
            reply = engine.get_reply(user_query)
            print(f"   Bot: {reply}")
