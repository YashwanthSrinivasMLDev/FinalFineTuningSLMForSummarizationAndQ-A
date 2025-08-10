import streamlit as st
# from main import create_retrieval_chain_custom
from main import start_main_app
st.title('FineTuning SLM')
#app info
st.markdown("###### FineTuning SLM")
# st.markdown("###### Your Text")
question = st.text_input("Type your question here")


# retrieval_chain = create_retrieval_chain_custom()
if question :
    start_main_app()
# print(response['answer'])
    st.header("Answer : ")
    # st.write(response['answer'])
