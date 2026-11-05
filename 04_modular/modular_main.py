# modular_main.py
# ---------------
# 04 단계: 분리된 모듈(프롬프트 · 모델 · 파서)을 가져와 체인으로 조립하고 실행하는 진입점입니다.
#
# 구성 파일
#   - modular_prompt.py : simple_prompt  (질문 형식을 정의한 프롬프트 템플릿)
#   - modular_llm.py    : chat_model     (설정이 끝난 Gemini 모델, .env의 API 키 사용)
#   - modular_parser.py : string_parser  (AI 응답에서 문자열만 추출하는 파서)
#
# 실행 방법: 04_modular 폴더에서  python modular_main.py
# (같은 폴더의 모듈을 import 하므로 이 폴더에서 실행해야 합니다.)

# 각 모듈에서 이미 만들어 둔 부품 객체를 가져옵니다.
# - modular_llm을 import 하는 순간 .env 로딩과 모델 객체 생성이 함께 일어납니다.
from modular_prompt import simple_prompt

from modular_llm import chat_model

from modular_parser import string_parser

import sys

# [출력 인코딩 보정] 터미널 출력 인코딩이 UTF-8이 아니면(예: 일부 Windows 콘솔) 한글이 깨질 수 있습니다.
# - sys.stdout.reconfigure()로 출력 인코딩을 UTF-8로 바꿔 한글/이모지를 정상 표시합니다.
if sys.stdout.encoding != 'utf-8':

    try:

        sys.stdout.reconfigure(encoding='utf-8')

    # reconfigure()를 지원하지 않는 오래된 Python 버전에서는 AttributeError가 나므로 무시하고 진행합니다.
    except AttributeError:

        pass

def main():

    # [1단계] LCEL 체인 조립: 프롬프트 -> 모델 -> 파서
    # - 파이프(|)로 연결하면 앞 단계의 출력이 다음 단계의 입력으로 자동 전달됩니다.
    # - 각 부품이 별도 파일로 분리되어 있어, 모델이나 프롬프트만 바꿔 끼우기 쉽습니다.
    modular_chain = simple_prompt | chat_model | string_parser

    # [2단계] 설명을 요청할 개념(질문 입력값) 지정
    # - 프롬프트 템플릿의 {concept} 변수에 대입될 값입니다.
    concept_query = "네트워크 라우터 (Router)"

    print(f"=== [메인 모듈 시작] 개념 설명 요청: '{concept_query}' ===")

    # 네트워크/인증/요금 오류가 나도 프로그램이 비정상 종료되지 않도록 try로 감쌉니다.
    try:

        # [3단계] 실제 전송될 프롬프트 미리보기
        # - format()은 변수만 채워 문자열로 보여줄 뿐 API 요청을 보내지 않습니다. (확인/디버깅용)
        formatted_prompt = simple_prompt.format(concept=concept_query)

        print(f"\n=== [실제 전송된 프롬프트] ===\n{formatted_prompt}\n=============================\n")

        # [4단계] 체인 실행
        # - invoke()에 딕셔너리로 변수를 넘기면 이때 실제 API 요청이 발생합니다.
        # - 키는 프롬프트의 변수명({concept})과 같아야 합니다.
        # - 파서를 거친 결과이므로 response는 AIMessage 객체가 아닌 순수 문자열입니다.
        response = modular_chain.invoke({"concept": concept_query})

        print("\n=== AI 설명 결과 ===")

        print(response)

    except Exception as e:

        # 흔한 원인: API 키 누락/오타(.env의 GOOGLE_API_KEY), 크레딧 소진(429), 네트워크 문제
        print(f"\n[오류 발생]: {e}")

        print("💡 04_modular 폴더와 연동되는 상위 혹은 로컬 가상 환경에 .env 키가 정상 기입되었는지 체크하세요.")

# 이 파일을 직접 실행할 때만 main()을 호출합니다. (다른 파일에서 import 하면 실행되지 않음)
if __name__ == "__main__":

    main()
