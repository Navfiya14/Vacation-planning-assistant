import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ==================================================
# CONFIGURATION
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "Vacation_Bot_INR_Dataset.csv")
DB_LOCATION = os.path.join(BASE_DIR, "chroma_trip_nexus_db")
COLLECTION_NAME = "vacation_planner_data"
BATCH_SIZE = 500

# ==================================================
# CHECK FILE EXISTS
# ==================================================
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"Dataset file not found.\nExpected at: {CSV_FILE}\n"
        "Make sure 'Vacation_Bot_INR_Dataset.csv' is inside your project folder."
    )

# ==================================================
# LOAD DATASET
# ==================================================
df = pd.read_csv(CSV_FILE)
df.fillna("", inplace=True)

print("Dataset Loaded Successfully")
print("Total rows:", len(df))

# ==================================================
# EMBEDDING MODEL
# ==================================================
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# ==================================================
# VECTOR STORE INITIALIZATION
# ==================================================
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

# ==================================================
# CHECK IF DB ALREADY HAS DATA
# ==================================================
existing_ids = vector_store.get(include=[])["ids"]
existing_count = len(existing_ids)

print("Existing documents in DB:", existing_count)

# ==================================================
# PREPARE DOCUMENTS
# ==================================================
documents = []
ids = []

for i, row in df.iterrows():

    content = (
        f"Destination: {row.get('Destination', '')}. "
        f"Location: {row.get('Location', '')}. "
        f"Resort Name: {row.get('Resort_Name', '')}. "
        f"Price per night: ₹{row.get('Price_INR', '')}. "
        f"Rating: {row.get('Rating', '')}. "
        f"Facilities: {row.get('Facilities', '')}. "
        f"Best Season: {row.get('Best_Season', '')}. "
        f"Description: {row.get('Description', '')}."
    )

    doc = Document(
        page_content=content,
        metadata={
            "destination": row.get("Destination", ""),
            "location": row.get("Location", ""),
            "price": row.get("Price_INR", ""),
            "rating": row.get("Rating", ""),
            "season": row.get("Best_Season", "")
        }
    )

    documents.append(doc)
    ids.append(str(i))

# ==================================================
# INGEST DATA (ONLY IF EMPTY)
# ==================================================
if existing_count == 0:
    print("Ingesting documents into Chroma...")

    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]

        vector_store.add_documents(
            documents=batch_docs,
            ids=batch_ids
        )

        print(f"Inserted {i} to {i + len(batch_docs)}")

    print("Vector DB successfully created!")
else:
    print("Using existing vector database. No re-ingestion needed.")

# ==================================================
# CREATE RETRIEVER
# ==================================================
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8}
)

print("TRIP NEXUS Vector DB Ready 🚀")