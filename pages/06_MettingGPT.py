import streamlit as st

st.set_page_config(
    page_tite="MeetingGPT",
    page_icon="👜"
)

# 사용자가 동영상을 올리면 오디오만 추출한 뒤, 오디오를 10분뒤로 분할(Whisper API가 최대 10분 길이의 파일 전송만 허용하기 때문)
# 10분 단위로 분할한 오디오 덩어리를 openAI API를 이용하여 Whisper 모델에 입력
# Whisper모델은 전체 대화를 받아적고 그 내용을 넘겨줌
# 넘겨받은 내용을 chain에 입력하여 전체 대화를 요약하게 하고, 문서를 embed한 뒤 
# 또 다른 chain에서 embed된 내용을 바탕으로 대화 내용에 대한 질문을 하도록 구성
# whisper 자체는 오픈소스이므로 무료이지만, openAI 플랫폼에 호스팅된 버전을 사용하려면 돈을 내야함(대신 속도가 빠름, 1분에 0.006달러)
