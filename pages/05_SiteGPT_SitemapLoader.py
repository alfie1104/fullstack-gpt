from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
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

llm = ChatOpenAI(
    temperature=0.1
)

answers_prompt = ChatPromptTemplate.from_template("""
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.
    If the answer answers the user question the score should be high, else it should be low.
    Make sure to always include the answer's score even if it's 0.
    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!
    Question: {question}
""")

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke({
    #         "question":question,
    #         "context":doc.page_content
    #     })
    #     answers.append(result.content)
    return [
        {
            "answer" : answers_chain.invoke({
                    "question":question,"context":doc.page_content
                }).content,
            "source": doc.metadata["source"],
            "date" : doc.metadata["lastmod"],
        } for doc in docs
    ]

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
        # Map Re Rank Chain
        # retriever로 부터 받은 document를 살펴보고 그 document들을 llm에게 전달하면서, 해당 document들로만 사용자의 question에 답변해달라고 요청함
        # 그리고 나서 각 document를 이용하여 답변들이 생성되면, llm에게 답변의 유용한 정도를 평가해달라고 요청함
        # 모든 답변과 점수는 또 다른 prompt에게 입력되고, 그 prompt는 다시 llm에 전달되어
        # '주어진 답변들 중 가장 높은 점수를 갖고 가장 최근에 작성된 것을 선택해줘'라고 요청
        # 따라서 2개의 chain이 필요함. 1) 답변 생성 및 채점 담당 / 2) 점수가 제일 높고 가장 최신정보를 담고 있는 답변을 고르는 chain

        # question이 retriever로 전달되어서 retriever는 docs를 반환하고,
        # 동일한 question이 다시 question 파라미터에 할당됨
        # retriever로부터 반환받은 docs와 question 파라미터를 get_answer 함수의 입력으로 전달하여
        # get_answers는 각 doc마다 답변 및 점수를 반환함
        chain = {
            "docs":retriever, 
            "question":RunnablePassthrough()
        } | RunnableLambda(get_answers) # docs와 question이 get_answers함수의 입력으로 전달됨
        # | RunnableLambda(choose_answer)

        chain.invoke("What is the pricing of GPT-4 Turbo with vision.")