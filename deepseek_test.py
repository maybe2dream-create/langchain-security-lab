"""DeepSeek API 키 연결 확인용.

실행: python deepseek_test.py   (langchain-lab 폴더, 가상환경 활성화 상태)

키는 이 파일에 쓰지 않는다. .env 의 DEEPSEEK_API_KEY 를 읽는다.
DeepSeek은 OpenAI와 같은 형식의 API를 제공하므로 langchain-openai의 ChatOpenAI에
DeepSeek 주소(base_url)만 바꿔서 연결한다. (Notion 02단계와 같은 방식)
"""
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 키 값은 출력하지 않고, 들어 있는지와 길이만 보여 준다.
if not api_key:
    print("[확인 실패] .env 에 DEEPSEEK_API_KEY 가 비어 있습니다.")
    raise SystemExit(1)
print(f"[확인] DEEPSEEK_API_KEY 읽음 (길이 {len(api_key)}자)")

model = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0,
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 AI 보안 교육 조교입니다. 한 문장으로 답하세요."),
    ("user", "{question}"),
])
chain = prompt | model | StrOutputParser()

try:
    print("답변:", chain.invoke({"question": "프롬프트 인젝션이 뭐야?"}))
except Exception as e:
    # 흔한 원인: 401 키 오류(복사 실수·공백), 402 잔액 부족(DeepSeek은 선불 충전 필요)
    print(f"[오류] {type(e).__name__}: {e}")
