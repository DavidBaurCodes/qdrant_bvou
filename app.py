# Notiz: Aktuell noch nicht fertig, Bitte noch Sidebar konfigurieren und annotieren.

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant   
from qdrant_client import QdrantClient
import json
import os
import tiktoken

# pip install langchain streamlit langchain-openai python-dotenv
# Zuweisung der Secrets direkt aus st.secrets
os.environ["OPENAI_API_KEY"] = st.secrets['openai_api_key']
qdrant_api_key = st.secrets("qdrant_api_key")
qdrant_url = st.secrets("qdrant_url")
qdrant_collection_name = st.secrets("qdrant_collection_name")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.full_responses = []

st.set_page_config(page_title="RAG - BVOU", page_icon="🔗", layout="wide")
st.title("BVOU Bot - QDRANT")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # Initialisiere die OpenAI-Embeddings
client = QdrantClient(qdrant_url, api_key=qdrant_api_key)
vectorstore = Qdrant(client=client, collection_name=qdrant_collection_name, embeddings=embeddings)  # Initialisiere den Qdrant-Client
enc = tiktoken.get_encoding("cl100k_base")

def tokenize_text(text):
    # Tokenisiere den Text mit dem spezifischen Encoder und zähle die Tokens
    tokenized_text = enc.encode(text)
    return len(tokenized_text)

def concatenate_text_lists(chat_history, context):
    # Kombiniere beide Listen
    combined_list = chat_history + context
    
    # Konvertiere jedes Element in einen String (falls einige keine Strings sind) und konkateniere sie
    combined_text = " ".join([str(item) for item in combined_list])
    
    return combined_text

def get_context_retriever_chain(vectorstore):
    # Initialisiere den Retriever und LLM (Language Model)
    llm = ChatOpenAI(model="gpt-4")
    retriever = vectorstore.as_retriever()  

    search_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Basierend auf dem vorherigen Gespräch, generiere eine Suchanfrage, um relevante Informationen für die Konversation zu erhalten")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, search_prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", '''Beantworte die folgende Frage basierend ausschließlich auf dem Dir vorgegebenem Kontext. 
         Du bist ein freundlicher Chatbot, auf einer Internetseite (www.orthinform.de) die orthopädische Informationen für 
         Patientinnen und Patienten bereitstellt und hast Wissen über den Inhalt der Website, sowie Fachliteratur. 
         Antworte höflich und hilfsbereit und erkläre Fachlicher medizinische Zusammenhänge in einer für Laien verständlichen Sprache. 
         Antworte in der Sprache in der Du gefragt wirst und bitte halte dich streng an den Dir vorgegebenen Kontext:\n\n{context}.'''),  
 
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Die Funktion 'get_response' definiert einen Generator, der die Antworten streamt
def get_response(query, chat_history):
    # Initialisiere deine Chains
    retriever_chain = get_context_retriever_chain(vectorstore)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    # Bereite deine Eingabedaten vor
    input_data = {
        "chat_history": chat_history,
        "input": query
    }
    
    # Rufe conversation_rag_chain.stream direkt mit input_data auf
    response_stream = conversation_rag_chain.stream(input_data)
    
    # Iteriere durch den Stream und speichere sowohl die Anfrage als auch die Antwort
    for response in response_stream:
        full_response_with_query = {"query": query, "response": response} 
        st.session_state.full_responses.append(full_response_with_query)
        # Gib nur den 'answer'-Teil für das Streaming zurück
        if 'answer' in response:
            yield response['answer']

def extract_context_and_metadata(response):
    context = response.get('context', [])
    chat_history = response.get('chat_history', [])
    combined_text = concatenate_text_lists(chat_history, [doc.page_content for doc in context])
    metadata = [doc.metadata for doc in context]
    page_content = [doc.page_content for doc in context]
    return combined_text, metadata, page_content


def get_source_info(metadata):
    return metadata.get('url', metadata.get('name', 'Unbekannte Quelle'))

def show_document_info(doc_info):
    try:
        if "page_content" in doc_info:
            with st.expander("Inhalt"):
                st.markdown(doc_info["page_content"])
        else:
            st.write(doc_info)
    except KeyError as e:
        st.error(f"Fehler beim Zugriff auf Dokumentattribut: {e}")
    except Exception as e:
        st.error(f"Unerwarteter Fehler: {str(e)}")

def show_sources(all_sources_info):
    if all_sources_info:
        st.markdown("Quellen:")
        for source_info in all_sources_info:
            st.write(f"- {source_info}")

total_token_length = 0

# Konversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# Benutzereingabe
user_query = st.chat_input("Deine Frage")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    # Direkte Verwendung von st.write_stream mit dem Generator
    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))
        
   
    st.session_state.chat_history.append(AIMessage(ai_response))   
 
    
# Nachdem die Antworten im Chatfenster verarbeitet und gestreamt wurden
if "full_responses" in st.session_state:
    st.sidebar.title("Extrahierte Texte")

    all_sources_info = []
    total_token_length = 0

    for response_with_query in st.session_state.full_responses:
        query = response_with_query["query"]
        response = response_with_query["response"]

        combined_text, metadata, page_content = extract_context_and_metadata(response)
        total_token_length += tokenize_text(combined_text)

        with st.sidebar.expander(f"Kontext: {query}", expanded=True):
            for doc_info in metadata:
                show_document_info(doc_info)
            for doc_info in page_content:
                show_document_info(doc_info)
 

        # Sammle Quelleninformationen
        for doc_metadata in metadata:
            source_info = get_source_info(doc_metadata)
            all_sources_info.append(source_info)
 

    # Anzeige der Gesamttokenanzahl
    st.write("Tokens:", total_token_length)
