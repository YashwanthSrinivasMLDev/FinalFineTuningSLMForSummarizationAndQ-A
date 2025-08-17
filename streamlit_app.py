import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import streamlit as st
from main import start_main_app
from TrainingSLM import run_fine_tuned_model
from  evaluation_of_model import evaluate_all_models_summary
import pandas as pd
st.title('FineTuning SLM')
#app info
# st.markdown("###### FineTuning SLM")
# st.markdown("###### Your Text")
st.set_page_config(layout="wide")
info_text = st.empty()

@st.cache_resource
def load_model():
    fine_tuned_model, tokenizer = start_main_app()

    print("starting the fine tuning app")
    return fine_tuned_model, tokenizer


fine_tuned_model, tokenizer= load_model()

if not fine_tuned_model:
    print("error in creating model : re-rerun the app")
else :
    article = st.text_input("Paste your article here for summarization")
    st.markdown("or")
    question = st.text_input("Ask your question here")
    st.markdown("or")
    st.markdown("Evaluate models on already existing test-data -- models being compared : microsoft/phi-2 (fine-tuned on insurance data) vs TinyLlama")
    # st.text("Choose an use-case for evaluating the test-cases   ")

    if article :
        print("generating output ")
        # info_text.text("generating output")
        info_text.markdown(
            '<p style="color: orange; font-size:18px;">app status : generating output</p>',
            unsafe_allow_html=True
        )
        output = run_fine_tuned_model(fine_tuned_model, article)
        # info_text.text("generation done")
        info_text.markdown(
            '<p style="color: green; font-size:18px;">app status : generation done</p>',
            unsafe_allow_html=True
        )
        # print(response['answer'])
        st.header("Output  : ")
        st.write(output)


def evaluate(use_case):
    print("inside evaluate , model use case " , use_case)
    # info_text.text("app status : evaluating models")
    info_text.markdown(
        '<p style="color: orange; font-size:18px;">app status : evaluating models</p>',
        unsafe_allow_html=True
    )
    results = evaluate_all_models_summary(use_case, fine_tuned_model, tokenizer )
    pandas_dataframe = pd.DataFrame(results)
    return pandas_dataframe
    # st.table(pandas_dataframe)
    print("clicked ")



model_use_case = ""

if "model_use_case" in st.session_state:
    st.session_state.model_use_case = ""

if st.button("Evaluate on summarization test-cases"):
    st.markdown(f"Selected use-case : Summarization")
    st.session_state.model_use_case="summarization"
    dataframe = evaluate("summarization")
    # st.table(dataframe)
    st.dataframe(dataframe, height=200, use_container_width=False)
    info_text.markdown(
        '<p style="color: green; font-size:18px;">app status : evaluating models for summarization</p>',
        unsafe_allow_html=True
    )


if st.button("Evaluate on Q&A test-cases"):
    st.markdown(f"Selected use-case : qa")
    st.session_state.model_use_case="qa"
    dataframe = evaluate("qa")
    # st.table(dataframe)
    st.dataframe(dataframe, height=200, use_container_width=False)
    info_text.markdown(
        '<p style="color: green; font-size:18px;">app status : evaluating models for qa</p>',
        unsafe_allow_html=True
    )

# if st.button("Evaluate test data"):
#     dataframe = evaluate(st.session_state.model_use_case)
#     st.table(dataframe)
#     info_text.markdown(
#         '<p style="color: green; font-size:18px;">app status : evaluating models</p>',
#         unsafe_allow_html=True
#     )
#     # info_text.text("evaluation done")