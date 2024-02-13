import streamlit as st
import subprocess # cmd 명령어를 파이썬코드에서 실행시켜줌
from pydub import AudioSegment # pydub을 이용하여 긴 audio파일을 여러개의 짧은 audio파일로 쪼갤 수 있음
import math
import openai
import glob # 디렉토리 안에서 특정한 이름을 갖는 파일을 찾을 수 있음

# 사용자가 동영상을 올리면 오디오만 추출한 뒤, 오디오를 10분단위로 분할(Whisper API가 최대 10분 길이의 파일 전송만 허용하기 때문)
# 10분 단위로 분할한 오디오 덩어리를 openAI API를 이용하여 Whisper 모델에 입력
# Whisper모델은 전체 대화를 받아적고 그 내용을 넘겨줌
# 넘겨받은 내용을 chain에 입력하여 전체 대화를 요약하게 하고, 문서를 embed한 뒤 
# 또 다른 chain에서 embed된 내용을 바탕으로 대화 내용에 대한 질문을 하도록 구성
# whisper 자체는 오픈소스이므로 무료이지만, openAI 플랫폼에 호스팅된 버전을 사용하려면 돈을 내야함(대신 속도가 빠름, 1분에 0.006달러)


def extract_audio_from_video(video_path, audio_path):
    # -i : 입력, -vn : 비디오 제거 (video nope)
    command = ["ffmpeg","-i",video_path,"-vn",audio_path]    
    subprocess.run(command)

def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    track = AudioSegment.from_mp3(audio_path)
    chunk_len = chunk_size * 60 * 1000 # ms단위 # chunk_size는 분 단위 시간을 입력
    chunks = math.ceil(len(track)/chunk_len)

    # audio파일을 특정 시간길이(chunk_len)로 잘라냄
    for i in range(chunks):
        start_time = i*chunk_len
        end_time = (i+1)*chunk_len
        chunk = track[start_time:end_time]
        chunk.export(f"{chunks_folder}/chunk_{i}.mp3",format="mp3")

def transcribe_chunks(chunk_folder, destination):
    files = glob.glob(f"{chunk_folder}/*.mp3")
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
        # file을 write binary 모드로 오픈
        video_content = video.read()
        video_path = f"./.cache/{video.name}"
        with open(video_path,"wb") as f:
            f.write(video_content)
        extract_audio_from_video(video_path, video_path.replace("mp4","mp3")) # 비디오에서 오디오만 추출하여 xxxx.mp3에 저장
