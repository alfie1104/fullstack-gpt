from langchain.prompts import ChatPromptTemplate
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.embeddings import OllamaEmbeddings, CacheBackedEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import os

st.set_page_config(
    page_title="PrivateGPT",
    page_icon="📃",
)

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    # *args : 무수히 많은 args를 받을 수 있음 (예 : on_llm_start(1,2,3,4, ...))
    # **kwargs : 무수히 많은 keyword arguments를 받을 수 있음 (예 : on_llm_start(a=1, b=4, ...))
    def on_llm_start(self, *args, **kwargs):
        # llm이 시작되면 message_box라고 이름붙인 빈 공간을 만듦
        self.message_box = st.empty()
    
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
    
    def on_llm_new_token(self, token, *args, **kwargs):
        # token : chain에서 streaming되는 글자들
        # 새로운 token이 생성될때마다 message_box에 토큰을 추가함
        self.message += token
        self.message_box.markdown(self.message)
        

llm = ChatOllama(
    model="mistral:latest",
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(), #chain이 invoke될때 호출됨
    ]
)

# st.cache_data 데코레이터를 이용해서 파라미터에 어떤 변화가 있지 않으면 아래 함수를 재실행하지 않고 cache의 결과를 가져오도록 함
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_path,"wb") as f:
        f.write(file_content)

    cache_path = f"./.cache/private_embeddings/{file.name}"
    if  not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    cache_dir = LocalFileStore(cache_path)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OllamaEmbeddings(
        model="mistral:latest"
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def save_message(message, role):
    st.session_state["messages"].append({"message":message, "role":role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the following context and not your training data. If you don't know the answer
    just say you are don't know. DON'T make anything up.

    Context: {context}
    Question:{question}
    """
)


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

        with st.chat_message("ai"):        
            # ChatCallbackHandler클래스는 llm의 callback으로 등록되기때문에 llm에서 호출됨
            # 한편, chain은 with st.chat_message("ai") 구문에서 invoke되므로
            # ChatCallbackHander클래스에서 사용되는 st.write는  "ai" 메시지형태로 출력되게 됨
            chain.invoke(message) # 사용자가 보내는 message가 retriever의 파라미터, 그리고 question의 값으로 들어감(RunnablePassthrough)

else:
    st.session_state["messages"] = []