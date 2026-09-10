import os
import time

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai


# =========================
# 1. LOAD API KEY
# =========================

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in .env file.")
    st.stop()

client = genai.Client(api_key=API_KEY)


# =========================
# 2. PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="Student FAQ RAG",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student FAQ RAG")
st.write("Ask questions from your Student FAQ PDF documents.")


# =========================
# 3. EMBEDDING MODEL
# =========================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# =========================
# 4. CHROMADB
# =========================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="student_faq"
)


# =========================
# 5. EXTRACT PDF TEXT
# =========================

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append({
                "text": text,
                "page": page_number
            })

    return pages


# =========================
# 6. CREATE TEXT CHUNKS
# =========================

def create_chunks(
    pages,
    chunk_size=800,
    overlap=100
):

    chunks = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end]

            if chunk.strip():

                chunks.append({
                    "text": chunk,
                    "page": page_number
                })

            start += chunk_size - overlap

    return chunks


# =========================
# 7. UPLOAD PDF
# =========================

uploaded_file = st.file_uploader(
    "📄 Upload Student FAQ PDF",
    type=["pdf"]
)


# =========================
# 8. PROCESS PDF
# =========================

if uploaded_file:

    if st.button("Process FAQ"):

        with st.spinner("Processing PDF..."):

            pages = extract_text(uploaded_file)

            if not pages:

                st.error(
                    "❌ Could not extract text from this PDF."
                )
                st.stop()

            chunks = create_chunks(pages)

            texts = [
                chunk["text"]
                for chunk in chunks
            ]

            embeddings = embedding_model.encode(
                texts
            ).tolist()

            ids = [
                f"{uploaded_file.name}_{i}"
                for i in range(len(chunks))
            ]

            metadatas = [
                {
                    "source": uploaded_file.name,
                    "page": chunk["page"]
                }
                for chunk in chunks
            ]

            collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

        st.success(
            f"✅ Processed {len(chunks)} FAQ chunks."
        )


# =========================
# 9. ASK QUESTION
# =========================

question = st.text_input(
    "❓ Ask your question"
)


if st.button("Ask"):

    if not question.strip():

        st.warning("Please enter a question.")

    elif collection.count() == 0:

        st.warning(
            "Please upload and process a FAQ PDF first."
        )

    else:

        with st.spinner(
            "🔎 Searching FAQ and generating answer..."
        ):

            # Convert question into embedding
            question_embedding = embedding_model.encode(
                [question]
            ).tolist()

            # Search ChromaDB
            results = collection.query(
                query_embeddings=question_embedding,
                n_results=4
            )

            retrieved_docs = results["documents"][0]

            retrieved_metadata = results["metadatas"][0]

            # Combine retrieved chunks
            context = "\n\n".join(
                retrieved_docs
            )

            # =========================
            # GEMINI PROMPT
            # =========================

            prompt = f"""
You are a Student FAQ assistant.

Answer the student's question using ONLY
the information provided in the context.

Do not use outside knowledge.

If the answer is not available in the context,
say exactly:

"The information is not available in the FAQ."

Give a clear and concise answer.

CONTEXT:
{context}

STUDENT QUESTION:
{question}
"""


            # =========================
            # GEMINI RETRY SYSTEM
            # =========================

            models = [
                "gemini-3.8-flash",
                "gemini-3.7-flash",
                "gemini-3.6-flash"
            ]

            response = None

            for model in models:

                for attempt in range(3):

                    try:

                        response = client.models.generate_content(
                            model=model,
                            contents=prompt
                        )

                        if response:
                            break

                    except Exception as e:

                        error_message = str(e)

                        if (
                            "503" in error_message
                            or "UNAVAILABLE" in error_message
                            or "429" in error_message
                        ):

                            time.sleep(3)

                        else:

                            st.error(
                                f"❌ Gemini error: {error_message}"
                            )

                            response = None
                            break

                if response:
                    break


            # =========================
            # DISPLAY ANSWER
            # =========================

            if response:

                st.subheader("💡 Answer")

                st.write(
                    response.text
                )


                # =========================
                # DISPLAY SOURCES
                # =========================

                st.subheader("📚 Sources")

                shown_sources = set()

                for metadata in retrieved_metadata:

                    source = (
                        metadata["source"],
                        metadata["page"]
                    )

                    if source not in shown_sources:

                        st.write(
                            f"📄 {metadata['source']} "
                            f"— Page {metadata['page']}"
                        )

                        shown_sources.add(source)

            else:

                st.error(
                    "❌ Gemini is temporarily unavailable. "
                    "Please try again later."
                )