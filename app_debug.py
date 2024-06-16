import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
import os
import tiktoken
import toml

# Zuweisung der Secrets direkt aus st.secrets
os.environ["OPENAI_API_KEY"] = st.secrets['openai_api_key']
qdrant_api_key = st.secrets["qdrant_api_key"]
qdrant_url = st.secrets["qdrant_url"]
qdrant_collection_name = st.secrets["qdrant_collection_name"]

# Initialisieren des Chatverlaufs und der vollständigen Antworten im Sitzungsstatus, falls nicht vorhanden
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "full_responses" not in st.session_state:
    st.session_state.full_responses = []
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

# Konfigurieren der Streamlit-Seite
st.set_page_config(page_title="Orthinform Chatbot", page_icon="🔗", layout="wide")
st.title("Orthinform - Chatbot")

# Initialisieren der Embeddings und des Qdrant-Clients
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
client = QdrantClient(qdrant_url, api_key=qdrant_api_key)
vectorstore = Qdrant(client=client, collection_name=qdrant_collection_name, embeddings=embeddings)
enc = tiktoken.get_encoding("o200k_base")

# Prompts aus Dateien laden
def load_prompt(filename):
    with open(filename, 'r') as file:
        return file.read().strip()

system_prompt = load_prompt("system_prompt.txt")
search_prompt = load_prompt("search_prompt.txt")

# Funktion zum Tokenisieren des Textes
def tokenize_text(text):
    tokenized_text = enc.encode(text)
    return len(tokenized_text)

# Funktion zum Abrufen der Context-Retriever-Kette
def get_context_retriever_chain(vectorstore, k=5):
    llm = ChatOpenAI(model="gpt-4o")
    retriever = vectorstore.as_retriever(return_full_document=True, search_kwargs={"k": k})

    search_prompt_template = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", search_prompt)
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, search_prompt_template)
    return retriever_chain

# Funktion zum Erstellen der Conversational RAG-Kette
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Funktion zum Abrufen der Antwort und der Dokumente
def get_response_and_documents(query, chat_history):
    retriever_chain = get_context_retriever_chain(vectorstore)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    input_data = {
        "chat_history": chat_history,
        "input": query
    }
    
    response_stream = conversation_rag_chain.stream(input_data)
    documents = retriever_chain.invoke(input_data)  # Abrufen der Dokumente
    
    ai_response = ""
    for response in response_stream:
        full_response_with_query = {"query": query, "response": response}
        st.session_state.full_responses.append(full_response_with_query)
        if 'answer' in response:
            ai_response += response['answer']
            yield response['answer']
    
    st.session_state.retrieved_docs.append({"query": query, "documents": documents})
    
    return ai_response, documents

# Funktion zum Berechnen des Tokenverbrauchs des gesamten Prozesses
def calculate_total_tokens(user_query, chat_history, ai_response, documents):
    full_text = ""
    for message in chat_history:
        full_text += message.content + " "
    full_text += user_query + " "
    full_text += ai_response + " "  # AI-Response hinzufügen
    
    for doc in documents:
        full_text += doc.page_content + " "
    
    total_tokens = tokenize_text(full_text)
    return total_tokens

# Funktion zum Abrufen der URLs aus den Metadaten
def get_urls_from_metadata(documents):
    urls = set()
    for doc in documents:
        if "url" in doc.metadata:
            url_list = doc.metadata["url"]
            if isinstance(url_list, list):
                urls.update(url_list)
            else:
                urls.add(url_list)
    # Filtern von leeren und ungültigen URLs
    valid_urls = [url for url in urls if url and url.lower() not in ["", "n/a"] and len(url) >= 5]
    return valid_urls

# Anzeigen der Konversation im Hauptfenster
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Benutzereingabe für die Frage
user_query = st.chat_input("Deine Frage")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Verwenden des Generators für die KI-Antwort
    ai_response_content = ""
    documents = None
    with st.chat_message("AI"):
        ai_response_generator = get_response_and_documents(user_query, st.session_state.chat_history)
        for response in st.write_stream(ai_response_generator):
            ai_response_content += response  # Sammeln des gesamten AI-Responses
    
    # Abrufen der vollständigen Dokumente
    documents = st.session_state.retrieved_docs[-1]["documents"]
    st.session_state.chat_history.append(AIMessage(content=ai_response_content))

    # URLs in den Metadaten prüfen und am Ende der AI-Nachricht anzeigen
    urls = get_urls_from_metadata(documents)
    if urls:
        links_content = "<p style='font-size: medium;'>Links für weitere Informationen:</p>"
        for url in urls:
            links_content += f"<p style='font-size: small;'>- <a href='{url}' target='_blank'>{url}</a></p>"
        st.markdown(links_content, unsafe_allow_html=True)

    # Berechnen und Anzeigen des gesamten Tokenverbrauchs
    total_tokens = calculate_total_tokens(user_query, st.session_state.chat_history, ai_response_content, documents)
    st.session_state.retrieved_docs[-1]["tokens"] = total_tokens

# Anzeigen der abgerufenen Dokumente in Expandern in der Seitenleiste
st.sidebar.title("Metadaten - Tokens")
if st.session_state.retrieved_docs:
    for entry in st.session_state.retrieved_docs:
        query = entry["query"]
        documents = entry["documents"]
        tokens = entry["tokens"]
        with st.sidebar.expander(f"Dokumente für die Anfrage: {query} (Tokens: {tokens})", expanded=False):  # Default to collapsed
            for doc in documents:
                st.write("Metadata:", doc.metadata)
                st.write("Page Content:", doc.page_content)
else:
    st.sidebar.write("Hier werden Dokumente angezeigt, die während der Konversation abgerufen wurden.")
