import os
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import BaseOutputParser
import json


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```","").replace("json","")
        # json문자열에서 필요없는 문자를 제거한 뒤 파이썬 객체로 만듦
        # ```json ~~ ````이 들어간 이유 : formatting_prompt에서 Example설명시 json데이터의 도입부는 ```json로 시작하라고 했기 때문
        # ```json으로 답변을 시작하고 ```로 끝낸다고 예시를 보여주면, AI가 "물론이지요, 기꺼이 도와줄게요! 와 같은 문구를 생략하고 바로 json형태의 데이터를 반환함"
        return json.loads(text)
    
output_parser = JsonOutputParser()


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓"
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106", # context window의 크기가 16,385 tokens이므로 한번에 많은 내용을 처리할 수 있음(가격 : 1k토큰당 $0.001)
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ]
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

questions_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                    You are a helpful assistant that is role playing as a teacher.
                        
                    Based ONLY on the following context make 10 questions to test the user's knowledge about the text.
                    
                    Each question should have 4 answers, three of them must be incorrect and one should be correct.
                        
                    Use (o) to signal the correct answer.
                        
                    Question examples:
                        
                    Question: What is the color of the ocean?
                    Answers: Red|Yellow|Green|Blue(o)
                        
                    Question: What is the capital or Georgia?
                    Answers: Baku|Tbilisi(o)|Manila|Beirut
                        
                    Question: When was Avatar released?
                    Answers: 2007|2001|2009(o)|1998
                        
                    Question: Who was Julius Caesar?
                    Answers: A Roman Emperor(o)|Painter|Actor|Model
                        
                    Your turn!
                        
                    Context: {context}
                """,
            )
        ]
    )

questions_chain = {
        "context" : format_docs
    } | questions_prompt | llm

# 아래 template에서 {{ 을 사용한 이유 : { 를 사용하면 langchain은 변수를 할당하는 부분으로 생각하기 때문
formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are a powerful formatting algorithm.
                
                You format exam questions into JSON format.
                Answers with (o) are the correct ones.
                
                Example Input:
                Question: What is the color of the ocean?
                Answers: Red|Yellow|Green|Blue(o)
                    
                Question: What is the capital or Georgia?
                Answers: Baku|Tbilisi(o)|Manila|Beirut
                    
                Question: When was Avatar released?
                Answers: 2007|2001|2009(o)|1998
                    
                Question: Who was Julius Caesar?
                Answers: A Roman Emperor(o)|Painter|Actor|Model
                
                
                Example Output:
                
                ```json
                {{ "questions": [
                        {{
                            "question": "What is the color of the ocean?",
                            "answers": [
                                    {{
                                        "answer": "Red",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Yellow",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Green",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Blue",
                                        "correct": true
                                    }},
                            ]
                        }},
                                    {{
                            "question": "What is the capital or Georgia?",
                            "answers": [
                                    {{
                                        "answer": "Baku",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Tbilisi",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "Manila",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Beirut",
                                        "correct": false
                                    }},
                            ]
                        }},
                                    {{
                            "question": "When was Avatar released?",
                            "answers": [
                                    {{
                                        "answer": "2007",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "2001",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "2009",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "1998",
                                        "correct": false
                                    }},
                            ]
                        }},
                        {{
                            "question": "Who was Julius Caesar?",
                            "answers": [
                                    {{
                                        "answer": "A Roman Emperor",
                                        "correct": true
                                    }},
                                    {{
                                        "answer": "Painter",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Actor",
                                        "correct": false
                                    }},
                                    {{
                                        "answer": "Model",
                                        "correct": false
                                    }},
                            ]
                        }}
                    ]
                }}
                ```
                Your turn!
                Questions: {context}
            """,
        )
    ]
)

formatting_chain = formatting_prompt | llm

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

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    """ 
     파라미터명에 _를 붙인 이유 : streamlit이 파라미터를 해싱해서 서명을 만든 뒤 서명 값을 비교해서 파라미터가 변경되었는지 인식하는데
     document데이터를 이용해서 서명을 만들 수 없으므로 _를 붙여서 streamlit에게 서명을 만들지 말라고 안내한 것임
     그러면 이제 docs내용이 변경되어도 함수가 재실행되지 않는 문제가 발생함
     그래서 topic이라는 추가 변수를 설정하여 topic이 바뀌면 함수가 실행되도록 설정하였음
    """
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(
                top_k_results=5, 
                # lang="ko" #한글 문서를 검색하고 싶은 경우
                )
    docs = retriever.get_relevant_documents(term)
    return docs

with st.sidebar:
    docs = None
    choice = st.selectbox("Choose what you want to use.", (
        "File","Wikipedia Article"
    ))
    if choice == "File":
        file = st.file_uploader("Upload a .docs, .txt or .pdf file", type=["pdf","txt","docx"])

        if file:
            docs = split_file(file)
            
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.

    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.

    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    start = st.button("Generate Quiz")

    if start:
        response = run_quiz_chain(docs, topic if topic else file.name)
        st.write(response)
