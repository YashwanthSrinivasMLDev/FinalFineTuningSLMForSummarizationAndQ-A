import streamlit as st
# from main import create_retrieval_chain_custom
from main import start_main_app
from TrainingSLM import test_model_after_fine_tuning

st.title('FineTuning SLM')
#app info
st.markdown("###### FineTuning SLM")
# st.markdown("###### Your Text")


@st.cache_resource
def load_model():
    fine_tuned_model = start_main_app()

    print("starting the fine tuning app")
    return fine_tuned_model


fine_tuned_model= load_model()

if not fine_tuned_model:
    print("error in creating model : re-rerun the app")
else :
    question = st.text_input("Type your question here")

    if question :
        print("generting output ")
        output = test_model_after_fine_tuning(fine_tuned_model, question)
        # print(response['answer'])
        st.header("Output  : ")
        st.write(output)
