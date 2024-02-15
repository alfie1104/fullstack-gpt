import streamlit as st
import subprocess # cmd 명령어를 파이썬코드에서 실행시켜줌
from pydub import AudioSegment # pydub을 이용하여 긴 audio파일을 여러개의 짧은 audio파일로 쪼갤 수 있음
import math
import openai
import glob # 디렉토리 안에서 특정한 이름을 갖는 파일을 찾을 수 있음
import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore


# 사용자가 동영상을 올리면 오디오만 추출한 뒤, 오디오를 10분단위로 분할(Whisper API가 최대 10분 길이의 파일 전송만 허용하기 때문)
# 10분 단위로 분할한 오디오 덩어리를 openAI API를 이용하여 Whisper 모델에 입력
# Whisper모델은 전체 대화를 받아적고 그 내용을 넘겨줌
# 넘겨받은 내용을 chain에 입력하여 전체 대화를 요약하게 하고, 문서를 embed한 뒤 
# 또 다른 chain에서 embed된 내용을 바탕으로 대화 내용에 대한 질문을 하도록 구성
# whisper 자체는 오픈소스이므로 무료이지만, openAI 플랫폼에 호스팅된 버전을 사용하려면 돈을 내야함(대신 속도가 빠름, 1분에 0.006달러)

llm = ChatOpenAI(
    temperature=0.1
)

has_transcript = os.path.exists("./.cache/podcast.txt")

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=800,
    chunk_overlap=100,
)


# st.cache_data 데코레이터를 이용해서 파라미터에 어떤 변화가 있지 않으면 아래 함수를 재실행하지 않고 cache의 결과를 가져오도록 함
@st.cache_data(show_spinner="Embedding file...")
def embed_file(file_path):
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_path}")
    loader = TextLoader(transcript_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

@st.cache_data()
def extract_audio_from_video(video_path, audio_path):
    if has_transcript:
        return
    # -y: 동일한 파일이 있을 경우 덮어쓰기할지 여부, -i : 입력, -vn : 비디오 제거 (video nope)
    command = ["ffmpeg","-y","-i",video_path,"-vn",audio_path]    
    subprocess.run(command)

@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000 # ms단위 # chunk_size는 분 단위 시간을 입력
    chunks = math.ceil(len(track)/chunk_len)

    if not os.path.exists(chunks_folder):
        os.makedirs(chunks_folder)

    # audio파일을 특정 시간길이(chunk_len)로 잘라냄
    for i in range(chunks):
        start_time = i*chunk_len
        end_time = (i+1)*chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3",format="mp3")

@st.cache_data()
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"{chunk_folder}/*.mp3")
    # glob로 가져온 파일들이 이름순으로 정렬되어있다고 보장할 수 없기때문에 이름순으로 정렬해줘야함(audio순서대로 자막을 만들기 위해)
    files.sort()

    for file in files:
        # # openai.Audio.strascribe의 whisper 모델을 이용하여 오디오에서 텍스트를 추출할 수 있음
        # 파일 사용 뒤에 closed 되도록 with문을 이용해서 파일 오픈하겠음
        # rb : read as binary 모드, a: append 모드 (문자열을 하나의 파일에 계속 이어붙이기 위해 append모드로 열었음)
        with open(file,"rb") as audio_file, open(destination,"a") as text_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            text_file.write(transcript["text"])

st.set_page_config(
    page_title="MeetingGPT",
    page_icon="👜"
)

st.markdown("""
# MeetingGPT
            
Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask any questions about it.
Get started by uploading a video file in the sidebar.
"""
)

with st.sidebar:
    video = st.file_uploader("Video",type=["mp4","avi","mkv","mov"],)

if video:
    with st.status("Loading video...") as status:
        # file을 write binary 모드로 오픈
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4","mp3")
        chunks_folder = "./.cache/chunks"
        transcript_path = video_path.replace("mp4","txt")
        with open(video_path,"wb") as f:
            f.write(video_content)
        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path, audio_path) # 비디오에서 오디오만 추출하여 xxxx.mp3에 저장
        status.update(label="Cutting audio segments...")
        cut_audio_in_chunks(audio_path,10,chunks_folder)
        status.update(label="Transcribing audio...")
        transcribe_chunks(chunks_folder, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(["Transcript","Summary","Q&A"])

    with transcript_tab:
        with open(transcript_path, "r") as file:
            st.write(file.read())
    
    # summary_tab에서는 trascript파일을 Refined chain으로 요약하여 제공 (transcript파일을 여러개의 document로 쪼갠 다음 summary를 계속 업데이트)
    # Refined chain : input document들을 순회하면서 답변을 계속 업데이트(여러개의 document가 있을 때 첫번째 document로 첫번째 답변을 생성하고, 두번째 document와 첫번째 답변으로 개선된 답변을 생성, 반복...)
    with summary_tab:
        start = st.button("Generate summary")

        if start:
            loader = TextLoader(transcript_path)
            docs = loader.load_and_split(text_splitter=splitter)

            first_summary_prompt = ChatPromptTemplate.from_template("""
                Write a concise summary of the following:
                "{text}"
                CONCISE SUMMARY:
            """)

            first_summary_chain = first_summary_prompt | llm | StrOutputParser()

            summary = first_summary_chain.invoke({
                "text": docs[0].page_content
            })

            refine_prompt = ChatPromptTemplate.from_template(
                """
                Your job is to produce a final summary.
                We have provided an existing summary up to a certain point: {existing_summary}
                We have the opportunity to refine the existing summary (only if needed) with some more context below.
                ------------
                {context}
                ------------
                Given the new context, refine the original summary.
                If the context isn't useful, RETURN the original summary.
                """
            )

            refine_chain = refine_prompt | llm | StrOutputParser()

            with st.status("Summarizing...") as status:
                for i, doc in enumerate(docs[1:]):
                    status.update(label=f"Processing document {i+1}/{len(docs)-1}")
                    # 기존 summary와 새로운 document 를 합쳐서 요약한 뒤 summary를 업데이트
                    summary = refine_chain.invoke({
                        "existing_summary":summary,
                        "context":doc.page_content
                    })

            st.write(summary)
    
    with qa_tab:
        retriever = embed_file(transcript_path)
        docs = retriever.invoke("do they talk about marcus aurelius?")

        st.write(docs)