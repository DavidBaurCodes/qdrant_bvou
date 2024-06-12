import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
import toml
import os

# Zuweisung der Secrets direkt aus st.secrets
os.environ["OPENAI_API_KEY"] = st.secrets['openai_api_key']
qdrant_api_key = st.secrets["qdrant_api_key"]
qdrant_url = st.secrets["qdrant_url"]
qdrant_collection_name = st.secrets["qdrant_collection_name"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="RAG - BVOU", page_icon="🔗", layout="wide")
st.title("BVOU Bot - QDRANT")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
client = QdrantClient(qdrant_url, api_key=qdrant_api_key)
vectorstore = Qdrant(client=client, collection_name=qdrant_collection_name, embeddings=embeddings)

def get_context_retriever_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4o")
    retriever = vectorstore.as_retriever()

    search_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Basierend auf dem vorherigen Gespräch, generiere eine Suchanfrage, um relevante Informationen für die Konversation zu erhalten")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, search_prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(model="gpt-4o")
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

def get_response(query, chat_history):
    retriever_chain = get_context_retriever_chain(vectorstore)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    input_data = {
        "chat_history": chat_history,
        "input": query
    }
    
    response_stream = conversation_rag_chain.stream(input_data)
    
    for response in response_stream:
        if 'answer' in response:
            yield response['answer']

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

user_query = st.chat_input("Deine Frage")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))
        
    st.session_state.chat_history.append(AIMessage(content=ai_response))
