from langchain.document_loaders import SitemapLoader
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

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    loader = SitemapLoader(url)
    loader.requests_per_second = 5
    docs = loader.load()
    return docs
    

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
        docs = load_website(url)
        st.write(docs)