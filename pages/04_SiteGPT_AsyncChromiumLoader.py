from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import streamlit as st

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy()) # window에서 NotImplementedError가 발생하기 때문에 추가하였음


#  [AsyncChromiumLoader]
#  - AsyncChromiumLoader는 내부적으로 playwright 를 사용하기 때문에 최초 설치시 playwright instlal을 커맨드창에 입력해줘야함
#  - playwright가 사용되는 이유 ? playwright는 selenium과 비슷한 역할을 하지만 javascript를 사용하여 동적으로 생성되는 페이지의 정보를 가져올 수 있음
#  - AsyncChromiumLoader의 경우 여러개의 사이트에 대해서 html 문서를 가져오려고 할 경우, chromium browser가 여러개 열려야 하므로 속도가 느려질 수 있음
#  - Html2TextTransformer는 html형식의 문서에서 text만 뽑아내줌 (태그들을 없애줌)


st.set_page_config(
    page_title="SiteGPT",
    page_icon="💻"
)

html2text_transformer = Html2TextTransformer()

st.title("SiteGPT")


st.markdown("""
Ask questions about the content of a website.

Start by writing the URL of the website on the sidebar.
""")

with st.sidebar:
    url = st.text_input(
        "Write down a URL", 
        placeholder="https://example.com"
    )

if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    transformed = html2text_transformer.transform_documents(docs)
    st.write(docs)