import streamlit as st
# from main import create_retrieval_chain_custom
from main import retrieval_chain
st.title('Ethical ChatGPT')
#app info
st.markdown("###### ChatGPT but with the ethics of an animal rights activist")
# st.markdown("###### Your Text")
question = st.text_input("Type your question here")


# retrieval_chain = create_retrieval_chain_custom()
if question :
    response = retrieval_chain.invoke( {'input':question })
# print(response['answer'])
    st.header("Answer : ")
    st.write(response['answer'])
