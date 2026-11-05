"""langchain_apis.py 를 키 하드코딩 없이 실행하는 래퍼 (키 하드코딩 없이 실행하기 위해 추가).

02단계 원본은 키를 코드에 직접 적는 방식이다(일부러 취약하게 만든 단계).
실제 키를 파일에 남기지 않으려고, .env 의 값을 읽어 원본 모듈의 변수에 실행 중에만 넣어 준다.
.env 에 없는 키는 호출하지 않고 [건너뜀]으로 표시한다. (강사 실행 결과와 같은 표시)

실행: 02_apis 폴더에서  python run_with_env.py
"""
import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

import langchain_apis as apis

load_dotenv()

prompt = ChatPromptTemplate.from_template("{question}에 대해 한글로 친절히 답변해 주세요.")
user_query = "인공지능 개발 라이브러리인 '랭체인'이 왜 필요한지 1문장으로 요약해 줘."
print(f"공통 질문 내용: '{user_query}'")

targets = [
    ("GOOGLE_API_KEY", apis.test_google_gemini, "[1] Google Gemini"),
    ("OPENAI_API_KEY", apis.test_openai_gpt, "[2] OpenAI GPT"),
    ("DEEPSEEK_API_KEY", apis.test_deepseek, "[3] DeepSeek"),
]
for env_name, test, label in targets:
    key = os.getenv(env_name) or (os.getenv("GEMINI_API_KEY") if env_name == "GOOGLE_API_KEY" else None)
    if not key:
        print(f"\n=== {label} API 호출 테스트 ===\n[건너뜀]: .env에 {env_name}가 없습니다.")
        continue
    setattr(apis, env_name, key)   # 원본 모듈의 전역 변수에 실행 중에만 키를 넣는다 (파일에는 저장되지 않음)
    test(prompt, user_query)
