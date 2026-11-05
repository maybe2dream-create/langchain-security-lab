# gui_app.py
# ----------
# 05 단계: Streamlit 기반 웹 브라우저 채팅 인터페이스 구현
#
# 이 실습은 Streamlit 프레임워크를 이용해 Node.js나 JavaScript 없이
# 오직 파이썬 코드만으로 직관적인 대화형 웹 GUI를 구축하는 실습입니다.
#
# 실행 방법 (중요)
#   - Streamlit 앱은 `python gui_app.py`가 아니라 아래 명령으로 실행해야 합니다.
#       streamlit run gui_app.py
#   - `python gui_app.py`로 실행하면 "missing ScriptRunContext" 경고만 나오고 웹 화면이 뜨지 않습니다.
#
# 사전 준비
#   - .env 파일에 GOOGLE_API_KEY 또는 GEMINI_API_KEY 를 넣어 둡니다.

import os
import sys

import streamlit as st                                              # 파이썬만으로 웹 UI를 만드는 프레임워크
from dotenv import load_dotenv                                      # .env 파일의 값을 환경변수로 읽어오는 도구
from langchain_google_genai import ChatGoogleGenerativeAI           # Google Gemini 채팅 모델 래퍼
from langchain_core.prompts import ChatPromptTemplate               # 시스템 역할 + 사용자 입력을 묶는 프롬프트 템플릿
from langchain_core.output_parsers import StrOutputParser           # AIMessage에서 텍스트만 추출하는 파서

# Windows 터미널에서 한글 및 이모지 출력 시 인코딩 오류(UnicodeEncodeError) 방지
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# [1단계] .env 파일에서 구글 API 키 로드
# - 현재 파일 위치에서 상위 폴더 방향으로 .env를 찾아 읽습니다.
load_dotenv()

# [2단계] 웹 브라우저 기본 UI 디자인 및 헤더 정보 구성
# - set_page_config는 다른 st.* 호출보다 먼저(맨 처음) 실행해야 합니다.
st.set_page_config(page_title="간이 AI 챗봇", page_icon="💬")
st.title("💬 초간단 랭체인 챗봇 GUI")
st.write("Streamlit을 사용해 복잡한 프론트엔드 작업 없이 빠르게 웹 채팅 인터페이스를 구동합니다.")

# 브라우저 UI에서 시스템 프롬프트 템플릿을 시각화하여 학습을 돕습니다.
with st.expander("ℹ️ AI 시스템 프롬프트 확인"):
    st.code("당신은 무엇이든 상냥하게 답변해 주는 챗봇 도우미입니다.")

def get_api_key():
    # GOOGLE_API_KEY가 없으면 GEMINI_API_KEY를 대신 사용합니다. (둘 다 없으면 None)
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# [3단계] 랭체인 체인 로직 리소스 캐싱 처리
# - Streamlit은 UI 컴포넌트 이벤트가 발생할 때마다 파이썬 파일 전체를 처음부터 끝까지 재호출합니다.
# - 따라서 매번 중복해서 LLM 체인 객체를 재생성하지 않도록 `@st.cache_resource`로 메모리에 캐싱합니다.
@st.cache_resource
def get_llm_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-lite-latest",  # 원래 "gemini-2.5-flash" — 새 사용자에게 제공 중단(404)되어 2026-11-05 변경
        google_api_key=get_api_key(),
        temperature=0.7
    )
    # system: AI의 역할 지정 / user: 화면에서 입력받은 질문이 {topic} 자리에 들어갑니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 무엇이든 상냥하게 답변해 주는 챗봇 도우미입니다."),
        ("user", "{topic}")
    ])
    parser = StrOutputParser()
    # LCEL 체인: 프롬프트 -> 모델 -> 문자열 파서
    return prompt | llm | parser

# API 키가 없으면 모델 생성 단계에서 인증 오류가 나므로, 먼저 화면에 안내하고 실행을 멈춥니다.
if not get_api_key():
    st.error(".env 파일에 GOOGLE_API_KEY(또는 GEMINI_API_KEY)가 설정되어 있지 않습니다.")
    st.stop()

chain = get_llm_chain()

# [4단계] 대화 기록 유지 (세션 상태)
# - 스크립트가 매번 재실행되므로, 일반 변수는 초기화됩니다.
# - st.session_state에 저장한 값은 같은 브라우저 세션 동안 유지됩니다.
if "messages" not in st.session_state:
    st.session_state.messages = []

# 지금까지의 대화를 화면에 다시 그립니다. (재실행될 때마다 필요)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# [5단계] 사용자 입력 처리
# - st.chat_input: 화면 하단의 채팅 입력창. 엔터를 누르면 입력 문자열이 반환됩니다. (입력 전에는 None)
if user_input := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지를 기록하고 화면에 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # AI 답변 생성 (이때 실제 API 요청 발생)
    # - 이 체인은 이전 대화를 모델에 전달하지 않으므로, 매 질문을 독립적으로 처리합니다.
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중..."):
            try:
                response = chain.invoke({"topic": user_input})
            except Exception as e:
                response = f"[오류 발생]: {e}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
