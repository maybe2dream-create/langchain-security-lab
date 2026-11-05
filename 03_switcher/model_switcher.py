# model_switcher.py
# -----------------
# 03단계: 모델 제공자 동적 스위칭 (Google / OpenAI / DeepSeek)
#
# 02단계에서는 모델마다 함수를 따로 만들었지만, 여기서는 제공자 이름("google" 등) 하나만 바꿔서
# 같은 체인 코드(prompt | model | parser)에 서로 다른 LLM을 끼워 넣습니다.
# 소스 코드를 고치지 않고, 실행할 때 제공자를 고를 수 있습니다.
#
# 흐름도: model_switcher_flow.svg
#   ① 제공자 결정 → ② get_chat_model() → (③ 폴백 결합) → ④ ChatPromptTemplate → ⑤ LCEL 체인 → ⑥ invoke/출력
#   (아래 "▶ 플로우차트" 주석이 각 노드의 코드 위치를 가리킵니다.)
#
# 제공자 선택 우선순위 (위가 우선)
#   1) 실행 인자      : python model_switcher.py --provider deepseek
#   2) .env 변수      : LLM_PROVIDER=deepseek
#   3) 기본값         : google
#
# 폴백(선택): 선택한 모델이 실행 중 실패하면(예: 429 크레딧 소진, 인증 오류) 다른 모델로 자동 재시도합니다.
#   python model_switcher.py --provider google --fallback deepseek
#
# 사전 준비
#   - .env 파일에 사용할 제공자의 API 키를 넣어 둡니다. (키는 코드에 직접 쓰지 않습니다.)
#       GOOGLE_API_KEY=...     (provider="google", GEMINI_API_KEY 이름도 인식)
#       OPENAI_API_KEY=...     (provider="openai")
#       DEEPSEEK_API_KEY=...   (provider="deepseek")
#       LLM_PROVIDER=google    (선택: 기본 제공자)
#
# 실행 방법
# python model_switcher.py --provider google
# python model_switcher.py --provider deepseek
# python model_switcher.py --provider openai      # OPENAI_API_KEY 필요

import argparse
import os

from dotenv import load_dotenv                                   # .env 파일의 값을 환경변수로 읽어오는 도구
from langchain_core.prompts import ChatPromptTemplate            # 질문을 담아 모델에 전달할 프롬프트 템플릿
from langchain_core.output_parsers import StrOutputParser        # AIMessage에서 텍스트(content)만 추출하는 파서
from langchain_google_genai import ChatGoogleGenerativeAI        # Google Gemini 채팅 모델 래퍼
from langchain_openai import ChatOpenAI                          # OpenAI 채팅 모델 래퍼 (DeepSeek도 호환 API로 재사용)

# [환경설정] .env의 값을 환경변수로 로드합니다. (현재 파일 위치에서 시작해 상위 폴더 방향으로 .env를 탐색)
# - 이미 OS 환경변수로 설정된 값이 있으면 그 값이 우선합니다. (load_dotenv 기본 동작)
load_dotenv()

# 지원하는 제공자 목록: get_chat_model()의 분기와 --provider/--fallback 선택지가 함께 사용합니다.
PROVIDERS = ("google", "openai", "deepseek")
DEFAULT_PROVIDER = "google"


def get_api_key(*env_names: str) -> str:
    """환경변수에서 API 키를 읽어 돌려줍니다. 이름을 여러 개 주면 먼저 값이 있는 것을 사용합니다.

    - 모두 비어 있으면 어떤 키가 없는지 알려 주며 ValueError를 발생시킵니다.
    - 예) get_api_key("GOOGLE_API_KEY", "GEMINI_API_KEY") -> 둘 중 설정된 쪽의 값
    """
    for env_name in env_names:
        api_key = os.getenv(env_name)
        if api_key:
            return api_key
    # 첫 번째 이름을 대표 이름으로 안내합니다.
    raise ValueError(f"[오류] .env 파일에 {env_names[0]}가 설정되어 있지 않습니다.")


def get_chat_model(provider: str):
    """제공자 이름("google" / "openai" / "deepseek")에 맞는 LangChain 채팅 모델을 생성해 돌려줍니다.
    ▶ 플로우차트 노드 ② get_chat_model()

    - 대소문자는 구분하지 않습니다. ("Google"도 가능)
    - API 키가 없거나 지원하지 않는 제공자면 ValueError를 발생시킵니다.
    - temperature: 0에 가까울수록 일관된 답, 1에 가까울수록 다양한 답
    - 모델 객체만 만들 뿐 이 시점에는 API 요청을 보내지 않습니다. (요청은 chain.invoke() 때 발생)
    """
    provider = provider.lower()

    if provider == "google":
        return ChatGoogleGenerativeAI(
            model="gemini-flash-lite-latest",  # 원래 "gemini-2.5-flash" — 새 사용자에게 제공 중단(404)되어 2026-11-05 변경
            # .env의 키 이름이 GOOGLE_API_KEY 또는 GEMINI_API_KEY 어느 쪽이든 동작합니다.
            google_api_key=get_api_key("GOOGLE_API_KEY", "GEMINI_API_KEY"),
            temperature=0.7
        )

    elif provider == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=get_api_key("OPENAI_API_KEY"),
            temperature=0.7
        )

    elif provider == "deepseek":
        # DeepSeek은 OpenAI와 API 규격이 호환되므로 ChatOpenAI를 재사용하고,
        # base_url만 DeepSeek 서버로 바꿉니다.
        return ChatOpenAI(
            model="deepseek-chat",
            api_key=get_api_key("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
            temperature=0.7
        )

    else:
        raise ValueError(f"지원하지 않는 제공자입니다: {provider} (선택 가능: {', '.join(PROVIDERS)})")


def parse_args():
    """실행 인자를 읽습니다. 지정하지 않은 항목은 None으로 남습니다.

    - choices=PROVIDERS: 목록에 없는 이름을 주면 argparse가 사용법과 함께 바로 오류를 알려 줍니다.
    - 단, .env의 LLM_PROVIDER 값은 여기서 검사되지 않으므로 get_chat_model()이 검증합니다.
    """
    parser = argparse.ArgumentParser(description="LLM 제공자를 실행 시점에 전환하는 예제")
    parser.add_argument("--provider", choices=PROVIDERS,
                        help="사용할 제공자 (미지정 시 .env의 LLM_PROVIDER, 그것도 없으면 google)")
    parser.add_argument("--fallback", choices=PROVIDERS,
                        help="선택한 모델이 실행 중 실패하면 대신 사용할 제공자")
    return parser.parse_args()


def main():
    args = parse_args()

    # ① 제공자 결정  ▶ 플로우차트 노드 ① 제공자 결정
    # - 우선순위: 실행 인자 > .env의 LLM_PROVIDER > 기본값
    selected_provider = args.provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER
    print(f"[설정] 현재 활성화된 모델 제공자: {selected_provider.upper()}")

    # ② 모델 생성  ▶ 플로우차트 노드 ② get_chat_model()
    # - 키 누락·잘못된 제공자 이름 오류를 여기서 안내하고 종료합니다.
    try:
        model = get_chat_model(selected_provider)
    except Exception as e:
        print(f"\n{e}")
        print("💡 .env 파일에 해당 제공자의 API 키를 입력했는지, 제공자 이름이 올바른지 확인해 주세요.")
        return

    # ③ 폴백 결합(선택)  ▶ 플로우차트 노드 ③ with_fallbacks
    # - 기본 모델이 호출 중 실패하면 폴백 모델로 같은 입력을 자동 재시도하도록 묶습니다.
    # - 폴백 모델을 만들 수 없어도(키 누락 등) 기본 모델로는 계속 진행합니다.
    answer_label = selected_provider.upper()   # 답변 제목에 표시할 이름
    if args.fallback:
        if args.fallback.lower() == selected_provider.lower():
            print("[경고] 폴백이 기본 제공자와 같아 의미가 없으므로 무시합니다.")
        else:
            try:
                model = model.with_fallbacks([get_chat_model(args.fallback)])
                print(f"[설정] 폴백 모델 제공자: {args.fallback.upper()}")
                # 폴백이 실제로 답했는지는 알 수 없으므로(LangChain이 알려주지 않음) 제목에 "실패 시 폴백"으로 표기합니다.
                answer_label += f" (실패 시 {args.fallback.upper()} 폴백)"
            except Exception as e:
                print(f"[경고] 폴백 모델을 만들 수 없어 폴백 없이 진행합니다: {e}")

    # ④ 프롬프트 템플릿  ▶ 플로우차트 노드 ④ ChatPromptTemplate
    # - {word}는 실행 시점에 실제 단어로 치환됩니다.
    prompt = ChatPromptTemplate.from_template("다음 단어의 뜻을 한 줄로 알려줘: {word}")

    # ⑤ LCEL 체인 조립: 프롬프트 -> 모델 -> 문자열 파서  ▶ 플로우차트 노드 ⑤ LCEL 체인
    # - 모델이 무엇이든(폴백이 묶여 있든 아니든) 이 줄은 동일합니다.
    chain = prompt | model | StrOutputParser()

    # ⑥ 체인 실행 및 출력  ▶ 플로우차트 노드 ⑥ invoke → 출력
    # - 이때 실제 API 요청이 발생하므로 네트워크/인증/요금 오류가 날 수 있습니다.
    try:
        response = chain.invoke({"word": "추상화 (Abstraction)"})
        print(f"\n=== AI ({answer_label}) 답변 ===")
        print(response)
    except Exception as e:
        print(f"\n[실행 중 오류]: {e}")


if __name__ == "__main__":
    main()
