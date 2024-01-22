from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
import os

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

llm = ChatOpenAI(
    temperature=0.1,
)

# st.cache_data 데코레이터를 이용해서 파라미터에 어떤 변화가 있지 않으면 아래 함수를 재실행하지 않고 cache의 결과를 가져오도록 함
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_path,"wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message":message, "role":role})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """
    Answer the question using ONLY the following context. If you don't know the answer
    just say you are don't know. DON'T make anything up.

    Context: {context}
    """
    ),
    ("human","{question}")
])


st.title("DocumentGPT")

st.markdown("""
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
            
Upload your files on the sidebar.
""")

with st.sidebar:
    file = st.file_uploader("Upload a .txt .pdf or .docx file", type=["pdf","txt","docx"])

if file:
    retriever = embed_file(file)

    send_message("I'm ready! Ask away!","ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message,"human")
        # RunnableLambda는 chain과 그 내부 어디에서든 function을 호출할 수 있도록 해줌
        # retriever는 Langchain에 의해 호출됨 : retriever(질문)과 같은 방식으로 자동 호출됨
        chain = {
            "context" : retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | prompt | llm
        response = chain.invoke(message) # 사용자가 보내는 message가 retriever의 파라미터, 그리고 question의 값으로 들어감(RunnablePassthrough)
        send_message(response.content, "ai")

else:
    st.session_state["messages"] = []