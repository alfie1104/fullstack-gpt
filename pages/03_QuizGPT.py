import os
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler


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
            retriever = WikipediaRetriever(
                top_k_results=5, 
                # lang="ko" #한글 문서를 검색하고 싶은 경우
                )
            with st.status("Searching Wikipedia..."):
                docs = retriever.get_relevant_documents(topic)


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
        questions_response = questions_chain.invoke(docs)
        # chain에서 AI message를 받게 되는데 AI message에는 content가 있음
        st.write(questions_response.content)
        formatting_response = formatting_chain.invoke({
            "context":questions_response.content
        })
        st.write(formatting_response.content)