# modular_llm.py
# --------------
# 04 단계: 랭체인의 언어 모델(LLM) 컴포넌트를 독립 모듈로 분리 선언합니다.
#
# 프롬프트(modular_prompt.py) · 모델(이 파일) · 파서(modular_parser.py)를 파일별로 나눠 두면,
# 각 부품을 따로 수정·교체·재사용할 수 있고, modular_main.py에서는 import 한 뒤
# `simple_prompt | chat_model | string_parser` 처럼 파이프(|)로 조립하기만 하면 됩니다.
#
# 이 모듈이 외부에 제공하는 것
#   - chat_model : 설정이 끝난 Gemini 채팅 모델 객체 (modular_main.py가 import 해서 사용)
#
# 사전 준비
#   - .env 파일에 GOOGLE_API_KEY=... 를 넣어 둡니다. (키는 코드에 직접 쓰지 않습니다.)
#   - 키 변수 이름은 GOOGLE_API_KEY 또는 GEMINI_API_KEY 어느 쪽이든 인식합니다. (GOOGLE_API_KEY 우선)
#   - 둘 다 없으면 키가 None이 되어 Google 기본 인증(ADC)을 찾다가
#     "DefaultCredentialsError: Your default credentials were not found" 오류가 납니다.

import os                                                          # 환경변수를 읽기 위한 표준 라이브러리
from dotenv import load_dotenv                                     # .env 파일의 값을 환경변수로 읽어오는 도구
from langchain_google_genai import ChatGoogleGenerativeAI          # Google Gemini 채팅 모델 래퍼

# [1단계] 환경변수(.env) 로딩 수행
# - 인자 없이 호출하면 현재 파일 위치에서 시작해 상위 폴더 방향으로 .env를 찾아 읽습니다.
# - 이미 OS 환경변수로 설정된 값이 있으면 그 값이 우선합니다. (load_dotenv 기본 동작)
# - import 되는 순간 실행되므로, modular_main.py가 이 모듈을 import 할 때 함께 로드됩니다.
load_dotenv()

# [2단계] 환경변수에서 구글 API 키를 읽어와 안전하게 모델 인스턴스 구축
# - 소스 코드에 키가 하드코딩되지 않으므로, 이 모듈을 안심하고 다른 팀원들과 공유할 수 있습니다.
# - os.getenv()는 변수가 없으면 None을 돌려줍니다. 이 경우 모델 객체 생성 단계에서
#   DefaultCredentialsError가 나므로, 오류가 나면 .env의 변수 이름과 값을 먼저 확인하세요.
# - GOOGLE_API_KEY가 없으면 GEMINI_API_KEY를 대신 사용합니다. (`or`는 앞의 값이 비어 있을 때 뒤의 값을 선택)
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# [3단계] 채팅 모델 객체 생성 (이 시점에는 API 요청을 보내지 않고, 실제 요청은 chain.invoke() 때 발생)
# - model: 사용할 Gemini 모델명
# - google_api_key: 위에서 읽은 인증 키
# - temperature: 답변의 창의성 정도 (0.0에 가까울수록 일관되고 1.0에 가까울수록 창의적)
# - 모듈 최상위(전역)에 만들어 두었기 때문에, 여러 파일이 import 해도 같은 객체를 재사용합니다.
chat_model = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",  # 원래 "gemini-2.5-flash" — 새 사용자에게 제공 중단(404)되어 2026-11-05 변경
    google_api_key=api_key,
    temperature=0.7
)
