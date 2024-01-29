import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
import os


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

# st.cache_data 데코레이터를 이용해서 파라미터에 어떤 변화가 있지 않으면 아래 함수를 재실행하지 않고 cache의 결과를 가져오도록 함
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_path,"wb") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs # 파일을 읽어들여서 split한뒤 반환(임베딩은 하지 않았음)

with st.sidebar:
    choice = st.selectbox("Choose what you want to use.", (
        "File","Wikipedia Article"
    ))
    if choice == "File":
        file = st.file_uploader("Upload a .docs, .txt or .pdf file", type=["pdf","txt","docx"])

        if file:
            docs = split_file(file)
            st.write(docs)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            retriever = WikipediaRetriever(top_k_results=5)
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)