from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st

import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy()) # window에서 NotImplementedError가 발생하기 때문에 추가하였음

#  [SitemapLoader]
#  - sitemap.xml 파일을 가지고 있는 사이트에 대해서는 SitemapLoader를 이용하여 문서를 가져올 수 있다.
#  - AsyncChromiumLoader의 경우 여러개의 사이트에 대해서 html 문서를 가져오려고 할 경우, chromium browser가 여러개 열려야 하므로 속도가 느려질 수 있지만
#  - SitemapLoader는 chromium을 사용하지 않아서 위와 같은 문제가 없음
#  - 또한 AsyncChromiumLoader와 달리 SitemapLoader는 html 데이터를 정리해서 text만 뽑아옴(그런데 메뉴, 네비게이션 등의 텍스트들도 뽑아오기 때문에 필요없는 내용이 많음)
#  - 너무 빠르게 scrapping을 하면 차단될 수 있으므로 default 설정으로 SitemapLoader는 1초에 한번씩 요청을 보냄
#  - 속도르 더 줄이려면 SitemapLoader(url).requests_per_second 함수를 통해 조절할 수 있음
#  - SitemapLoader는 page로부터 모든 text를 추출하고 html을 제거하기 위해 내부적으로 beautiful soup(soup is just bunch of html) 를 사용하고 있음.
#  - parsing_function 속성을 설정하면 beautiful soup의 동작을 조정할 수 있음

def parse_page(soup):
    # beautiful soup가 생성하는 soup object를 이용해서 필요한 데이터만 추출

    # 다음과 같이 header와 footer 태그를 찾아서 soup에서 제거할 수 있음
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    # return soup.get_text() # header와 footer를 제거한 나머지 text를 반환
    
    # soup에서 반환한 text에서 필요없는 문자들을 제거
    return (
        str(soup.get_text())
        .replace("\n"," ")
        .replace("\xa0"," ") # \xa0 : non-breaking-space
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_data(show_spinner="Loading website...")
def load_website(url):    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200
    )

    loader = SitemapLoader(
        url 
        # , filter_urls=["https://openai.com/blog/data-partnerships"] # filter_urls를 이용하면 sitemap.xml에 있는 사이트 들 중 특정 사이트의 데이터만 가져올 수 있음
        # , filter_urls=[r"^(?!.*\/blog\/).*"] # 정규표현식을 이용해서 /blog/를 포함하지 않는 사이트만 가져옴
        # , filter_urls=[r"^(.*\/blog\/).*"] # 정규표현식을 이용해서 /blog/를 포함하는 사이트만 가져옴
        , parsing_function=parse_page # SitemapLoader가 내부적으로 사용하는 beautiful soup의 동작을 조정하기 위해서 parsing_function 속성을 사용할 수 있음
    )
    loader.requests_per_second = 5
    # docs = loader.load()
    docs = loader.load_and_split(text_splitter=splitter) # 긴 텍스트를 작게 잘라서 반환하도록 하기 위해 text_splittert사용
    # return docs

    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings()) # document를 검색에 사용하기 위해 임베딩 후 vector store에 저장
    return vector_store.as_retriever() # vector store를 chain에서 사용하기 위해 retriever(질문 쿼리에 적합한 document를 반환해줌)로 반환(retriever는 invoke 메소드를 가지고 있어서 chain에서 실행가능함)
    

st.set_page_config(
    page_title="SiteGPT",
    page_icon="💻"
)

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
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please wright down a Sitemap URL.")
    else:
        # docs = load_website(url)
        retriever = load_website(url) # 기존에는 load_website가 document를 그대로 반환했지만 이제 retriever로 반환
        docs = retriever.invoke("What is the price of GPT-4 Turbo")

        st.write(docs)