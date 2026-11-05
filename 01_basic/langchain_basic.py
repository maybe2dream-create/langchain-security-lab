# langchain_basic.py
# -----------------
# 01단계: 로컬 더미 모델로 작동하는 랭체인 (Fake Chat Model 기반)
#
# 이 실습은 실제 유료 LLM API 키를 사용하지 않고도, LangChain의 기본 구조와
# LCEL(LangChain Expression Language) 체인 연결 방식을 이해하도록 설계되었습니다.
#
# 흐름도: langchain_basic_flow.svg
#   ① 사용자 질문 → ② ChatPromptTemplate → ③ FakeListChatModel → ④ StrOutputParser → ⑤ 최종 답변 출력
#   (아래 "▶ 플로우차트" 주석이 각 노드의 코드 위치를 가리킵니다.)

from langchain_core.language_models import FakeListChatModel  # 테스트 목적으로 미리 정해진 답변을 내보내는 가짜 모델
from langchain_core.prompts import ChatPromptTemplate          # 사용자 입력을 받아 시스템 페르소나와 결합하는 프롬프트 템플릿
from langchain_core.output_parsers import StrOutputParser      # AI의 응답 객체(AIMessage)로부터 텍스트 내용만 추출하는 파서

def main():
    # [1단계]  Mock 또는 Fake LLM 인스턴스 생성  ▶ 플로우차트 노드 ③ FakeListChatModel
    # - responses 파라미터에 미리 정해진 답변들의 리스트를 전달합니다.
    # - 실제 API 요청을 보내지 않으므로 인터넷 연결이나 API 키가 필요 없습니다.
    model = FakeListChatModel(
        responses=[
            "안녕하세요! 저는 인터넷 연결 없이 동작하는 로컬 가짜(Fake) 모델입니다. 랭체인 연결 테스트에 성공했습니다!"
        ]
    )

    # [2단계] 프롬프트 템플릿(ChatPromptTemplate) 선언  ▶ 플로우차트 노드 ② ChatPromptTemplate
    # - system: AI의 역할 정의(페르소나 설정)
    # - user: 사용자의 입력을 동적으로 대입받을 변수 {user_input} 선언
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 랭체인 실습을 돕는 친절한 로봇 조우입니다."),
        ("user", "{user_input}")
    ])

    # [3단계] 출력 파서(Output Parser) 선언  ▶ 플로우차트 노드 ④ StrOutputParser
    # - LLM의 원본 출력물(AIMessage 객체)에서 순수한 텍스트 문자열(content)만 추출하는 도구입니다.
    parser = StrOutputParser()

    # [4단계] LCEL (LangChain Expression Language)을 통한 체인 조립  ▶ 플로우차트 ②→③→④ 연결(|)
    # - 파이프 연산자(|)를 사용하여 컴포넌트들을 연결합니다.
    # - 실행 흐름: 입력값 -> 프롬프트 템플릿 포맷팅 -> LLM 답변 생성 -> 문자열 파싱 -> 최종 텍스트 출력
    chain = prompt | model | parser

    # [5단계] 체인 실행 (invoke 메서드)  ▶ 플로우차트 노드 ① 사용자 질문 (query → invoke)
    # - 템플릿에 정의한 변수 'user_input'에 질문 문자열을 매핑하여 입력합니다.
    query = "안녕? 너는 누구야?"
    print(f"=== 사용자 질문: {query} ===")

    # chain.invoke()를 호출하면 데이터가 파이프(|)를 타고 순차적으로 처리됩니다.
    response = chain.invoke({"user_input": query})
    # ▶ 플로우차트 노드 ⑤ 최종 답변 출력
    print("\n=== AI 답변 ===")
    print(response)

if __name__ == "__main__":
    main()
