import streamlit as st 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma



embeddings = OpenAIEmbeddings( #← OpenAIEmbeddings를 초기화
    model="text-embedding-ada-002"
)

db = Chroma(
    persist_directory="./secure_data",
    embedding_function=embeddings
)


st.title("한화 무배당 상품요약서")
st.text("질문 입력: ")
user_input = st.text_area("input ", height=200)

# click 하면 동작
if st.button("입력"):
    if user_input:
        result = db.similarity_search("user_input")
        st.success("완료")
    else:
        st.warning("입력하세요")