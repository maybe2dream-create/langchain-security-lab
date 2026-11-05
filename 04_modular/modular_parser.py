# modular_parser.py
# -----------------
# 04 단계: 랭체인의 출력 파서 컴포넌트를 독립 모듈로 분리 선언합니다.
#
# 출력 형태를 바꾸고 싶을 때(예: 문자열 -> JSON) 이 파일의 파서만 교체하면 됩니다.
#
# 이 모듈이 외부에 제공하는 것
#   - string_parser : AI 응답에서 문자열만 추출하는 파서 (modular_main.py가 import 해서 사용)
#
# 체인에서의 위치: 프롬프트 -> 모델 -> [파서]   (이 파일은 맨 마지막 단계)

from langchain_core.output_parsers import StrOutputParser    # 모델 응답(AIMessage)에서 텍스트만 꺼내 주는 파서 클래스

# - string_parser: LLM의 응답 데이터에서 오직 텍스트 본문만 String 형태로 정제하는 컴포넌트
#
# 모델은 글자만이 아니라 메타데이터(토큰 사용량, 종료 사유 등)가 담긴 AIMessage 객체를 돌려줍니다.
#   - 파서를 거치지 않으면: AIMessage(content="...", response_metadata={...}, ...)
#   - StrOutputParser를 거치면: 그중 content(본문)만 담긴 순수 문자열 "..."
# 파서는 상태 없이 변환만 하므로, 이렇게 한 번 만들어 두고 여러 체인에서 재사용해도 안전합니다.
string_parser = StrOutputParser()
