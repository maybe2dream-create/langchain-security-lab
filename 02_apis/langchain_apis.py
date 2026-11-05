# langchain_apis.py
# -----------------
# 02 단계: 주요 LLM API 연동 실습 (Google Gemini, OpenAI GPT, DeepSeek API)
#
# 이 실습은 수강생들이 직접 발급받은 외부 플랫폼의 API 키들을 소스 코드에 기입하여
# 실제 클라우드 LLM 모델들(Gemini, GPT, DeepSeek)에 질의하고 답변 품질을 비교해봅니다.

from langchain_core.prompts import ChatPromptTemplate          # 공통으로 사용할 질문 프롬프트 템플릿
from langchain_core.output_parsers import StrOutputParser      # AI의 답변 객체에서 문자열만 필터링하는 파서
from langchain_google_genai import ChatGoogleGenerativeAI      # Google Gemini 연동 클래스
from langchain_openai import ChatOpenAI                        # OpenAI GPT 및 DeepSeek(호환) 연동 클래스

# =====================================================================
# [중요] 학생 교육 가이드용 API 키 설정부
# - 수강생분들은 아래 변수에 각 서비스(구글, 오픈AI, 딥시크)의 API 키를 입력하세요.
# - 주의: 실제 운영 환경에서는 보안을 위해 API 키를 절대 하드코딩하지 않습니다.
# =====================================================================
# (메모) 키 자리는 자리표시자다. 실제 키를 넣으면 이 파일은 커밋·공유하지 말 것.
#  키를 파일에 쓰지 않고 실행하려면 run_with_env.py 를 사용한다.
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
DEEPSEEK_API_KEY = "YOUR_DEEPSEEK_API_KEY"

def test_google_gemini(prompt_chain, query):
    print("\n=== [1] Google Gemini API 호출 테스트 ===")
    try:
        # ChatGoogleGenerativeAI 초기화
        # - model: 사용할 Gemini 모델명 (예: gemini-2.5-flash)
        # - google_api_key: 인증을 위한 구글 API 키 전달
        # - temperature: 답변의 창의성 정도 (0.0에 가까울수록 일관되고 1.0에 가까울수록 창의적)
        llm = ChatGoogleGenerativeAI(
            model="gemini-flash-lite-latest",  # 원래 "gemini-2.5-flash" — 새 사용자에게 제공 중단(404)되어 2026-11-05 변경
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7
        )
        # LCEL 체인 조립 및 실행 (프롬프트 | 모델 | 파서)
        chain = prompt_chain | llm | StrOutputParser()
        response = chain.invoke({"question": query})
        print(response)
    except Exception as e:
        print(f"[오류 발생]: {e}")

def test_openai_gpt(prompt_chain, query):
    print("\n=== [2] OpenAI GPT API 호출 테스트 ===")
    try:
        # ChatOpenAI 초기화
        # - model: 사용할 GPT 모델명 (예: gpt-4o-mini)
        # - api_key: 인증을 위한 오픈AI API 키 전달
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0.7
        )
        chain = prompt_chain | llm | StrOutputParser()
        response = chain.invoke({"question": query})
        print(response)
    except Exception as e:
        print(f"[오류 발생]: {e}")

def test_deepseek(prompt_chain, query):
    print("\n=== [3] DeepSeek API 호출 테스트 ===")
    try:
        # DeepSeek는 OpenAI API 규격을 그대로 호환하므로 ChatOpenAI 클래스를 재사용합니다.
        # - model: deepseek-chat (DeepSeek-V3/R1 등 서비스 모델)
        # - base_url: DeepSeek의 전용 엔드포인트 URL 지정 (https://api.deepseek.com/v1)
        # - api_key: 딥시크에서 발급받은 API 키 전달
        llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com/v1",
            temperature=0.7
        )
        chain = prompt_chain | llm | StrOutputParser()
        response = chain.invoke({"question": query})
        print(response)
    except Exception as e:
        print(f"[오류 발생]: {e}")

if __name__ == "__main__":
    # 공통 질문 프롬프트
    prompt = ChatPromptTemplate.from_template("{question}에 대해 한글로 친절히 답변해 주세요.")
    user_query = "인공지능 개발 라이브러리인 '랭체인'이 왜 필요한지 1문장으로 요약해 줘."

    print(f"공통 질문 내용: '{user_query}'")

    # 3사 API 테스트 순차 구동
    test_google_gemini(prompt, user_query)
    test_openai_gpt(prompt, user_query)
    test_deepseek(prompt, user_query)
