import os
import getpass
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI


# --- Set the API key as an environment variable ---
# Use the method you described. Replace "YOUR_API_KEY_HERE" with your actual key.
os.environ["GOOGLE_API_KEY"] = "YOUR-API-KEY" 

try:
    # # --- Load and process the document ---
    # print("Loading PDF document...")
    # file_path = Path(__file__).parent / "dictionary.pdf"
    # loader = PyPDFLoader(str(file_path)) # Ensure file_path is a string
    # docs = loader.load()
    # print("Document loaded.")

    # print("Splitting document into chunks...")
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200
    # )
    # split_docs = text_splitter.split_documents(docs)
    # print("Document split.")

    # print("First split chunk:", split_docs[0])
    # print("\nSplit docs length:", len(split_docs))
    # print("Original docs length:", len(docs))

    # # --- Create embeddings using the environment variable ---
    # print("\nInitializing the embedding model...")
    # # The library will now automatically find the GOOGLE_API_KEY from the environment
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    # print("Embedding model initialized.")

    # --- Generate and print the vector ---
    # print("Attempting to generate embedding vector ")
    # vector = embeddings.embed_query(split_docs[0].page_content)
    # print("\n--- SUCCESS! ---")
    # print("Vector:", vector)

    # # creating the vector store
    # vector_store = QdrantVectorStore.from_documents(
    #     documents=[],
    #     url="http://localhost:6333",
    #     collection_name="dictionary",
    #     embedding=embeddings
    # )
    # print("Vector store created.")
    # print("Adding documents to the vector store...")
    # vector_store.add_documents(documents=split_docs)
    # print("Documents added.")

    # creating the retriever
    print("Creating the retriever...")
    retriver = QdrantVectorStore.from_existing_collection(
        url="http://localhost:6333",
        collection_name="gen-ai-pdf",
        embedding=embeddings
    )
    print("Retriever created.")

    print("Querying the vector store...")
    query = "what were the responsibilities"
    relevant_chunks = retriver.similarity_search(query=query)
    print("Query result relevant chunks:", relevant_chunks)

    # now lets build the system prompt
    SYSTEM_PROMPT = f"""
    You are a helpful assistant that answers questions about the relevant chunks.
    Use the following pieces of context to answer the user's question.
    Context:
    {relevant_chunks}
    Answer the question in freindly manner like college boy in points .
    """
    print("System prompt created.")

    print("Generating response...")
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash-latest",
        temperature=0.3  # keep it low for factual answers
    )

    # Step 5: Ask the question
    print("Sending to LLM...")
    response = llm.invoke(SYSTEM_PROMPT, {"question": query})
    print("\n📘 Answer from LLM:")
    print(response.content)

    


except Exception as e:
    print(f"\n--- AN ERROR OCCURRED ---")
    print(f"Error details: {e}")
