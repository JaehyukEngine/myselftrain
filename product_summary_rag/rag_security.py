import streamlit as st 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage


template="""문장을 바탕으로 질문에 답하세요.

문장: 
{document}

질문: {query}
"""

embeddings = OpenAIEmbeddings( #← OpenAIEmbeddings를 초기화
    model="text-embedding-ada-002"
)

db = Chroma(
    persist_directory="./secure_data",
    embedding_function=embeddings
)

prompt = PromptTemplate(
    template=template,
    input_variables=['document', 'query']
)

chat = ChatOpenAI(
    model = "gpt-4o-mini-2024-07-18"
)

def qna(query, result):
    documents_string = " ".join([x.page_content for x in result])
    aimessage = chat([
        HumanMessage(content=prompt.format(document=documents_string, query=query))
    ])
    return aimessage.content

st.title("한화 무배당 상품요약서")
st.text("질문 입력: ")
user_input = st.text_area("input ", height=200)

# click 하면 동작
if st.button("입력"):
    if user_input:
        result = db.similarity_search(user_input)
        msg = qna(query=user_input, result=result)
        st.success(msg)
    else:
        st.warning("입력하세요")