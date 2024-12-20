import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import ollama

# Streamlit App Configuration
st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.title("📄 WhatsApp Chat Analyzer")
st.sidebar.title("Navigation")
st.sidebar.markdown("Use this app to analyze WhatsApp chat data, visualize relationships, and ask questions!")

# Utility: Load and Process File
def process_file(file_content):
    """
    Process uploaded file content for text analysis.
    """
    # Save content to a local file for loader compatibility
    with open("uploaded_chat.txt", "w", encoding="utf-8") as f:
        f.write(file_content)

    # Load text and split into chunks
    loader = TextLoader("uploaded_chat.txt")
    text_loader = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    return splitter.split_documents(text_loader)

# Utility: Extract Data Categories
def extract_categories(messages):
    """
    Extract useful categories such as links, timestamps, and quotes from messages.
    """
    categories = {"links": [], "timestamps": [], "quotes": [], "messages": []}

    for message in messages:
        text = message.page_content

        # Extract links, timestamps, and quotes
        categories["links"].extend(re.findall(r'(https?://[^\s]+)', text))
        categories["timestamps"].extend(re.findall(r'\[\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}:\d{2} (?:AM|PM)\]', text))
        categories["quotes"].extend(re.findall(r'"(.*?)"', text))
        categories["messages"].append(text)

    return categories

# Utility: Visualize Relationships
def visualize_relationships(categories):
    """
    Generate a graph to show relationships between extracted categories.
    """
    G = nx.Graph()
    for category in categories:
        G.add_node(category)

    for message in categories["messages"]:
        if any(link in message for link in categories["links"]):
            G.add_edge("messages", "links")
        if any(quote in message for quote in categories["quotes"]):
            G.add_edge("messages", "quotes")

    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=3000, font_size=12)
    plt.title("Content Relationship Mapping")
    st.pyplot(plt)

# Utility: Detect Chat Theme with LLaMA
def detect_chat_theme_llama(messages):
    """
    Detect the main themes of the chat using LLaMA 3.2 via Ollama API.
    """
    # Combine messages into a readable format
    chat_content = "\n".join(messages)

    # Define the question to detect themes
    question = """
    Analyze the chat messages and identify overarching themes or topics discussed. 
    Provide concise themes with brief explanations for each.
    """

    # Format the prompt with context
    formatted_prompt = f"Question: {question}\n\nContext:\n{chat_content}"

    # Query the LLaMA 3.2 model
    try:
        response = ollama.chat(
            model="llama3.2", 
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        # Extract and return themes from the response
        return response.get("message", {}).get("content", "No themes detected.")
    except Exception as e:
        return f"Error detecting themes: {e}"

# Utility: Detect Patterns
def detect_patterns(messages):
    """
    Detect the most common word patterns in messages.
    """
    all_text = " ".join(messages).split()
    word_frequencies = Counter(all_text)
    common_words = {"the", "and", "a", "to", "of", "is", "it", "in", "on", "that", "this"}
    filtered_frequencies = {word: count for word, count in word_frequencies.items() if word.lower() not in common_words}
    return sorted(filtered_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]

# Utility: RAG Chain
def run_rag_chain(question, messages, retriever):
    """
    Retrieve and generate answers using LangChain and Ollama LLM.
    """
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def ollama_llm(question, context):
        formatted_prompt = f"Question: {question}\n\nContext: {context}"
        response = ollama.chat(model="llama3.2", messages=[{'role': 'user', 'content': formatted_prompt}])
        return response['message']['content']

    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Main Application Workflow
uploaded_file = st.file_uploader("Upload a WhatsApp Chat File (TXT)", type="txt")

if uploaded_file:
    try:
        # Read file content and process messages
        file_content = uploaded_file.getvalue().decode("utf-8")
        messages = process_file(file_content)

        # Extract categories
        categories = extract_categories(messages)

        # Display extracted categories
        st.subheader("Extracted Categories")
        st.write("**Links Found:**", categories["links"] or "No links found.")
        st.write("**Timestamps Found:**", categories["timestamps"] or "No timestamps found.")
        st.write("**Quotes Found:**", categories["quotes"] or "No quotes found.")

        # Display graph visualization
        st.subheader("Content Relationship Graph")
        visualize_relationships(categories)

        # Display top word patterns
        st.subheader("Top Word Patterns")
        patterns = detect_patterns(categories["messages"])
        st.write(patterns)

        # Detect and display themes using LLaMA
        st.subheader("Detected Themes")
        themes = detect_chat_theme_llama(categories["messages"])
        st.write(themes)

        # RAG Question-Answering
        embeddings = OllamaEmbeddings(model="llama3.2")
        vector_store = Chroma.from_documents(documents=messages, embedding=embeddings)
        retriever = vector_store.as_retriever()

        # question = st.text_input("Ask a question about the chat content:")
        # if question:
        #     st.subheader("Generated Answer")
        #     result = run_rag_chain(question, messages, retriever)
        #     st.write(result)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a WhatsApp chat file to get started.")
