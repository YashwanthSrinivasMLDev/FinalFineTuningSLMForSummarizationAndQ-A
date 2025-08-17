import streamlit as st
from main import start_main_app
from TrainingSLM import run_fine_tuned_model
from  evaluation_of_model import evaluate_all_models_summary
import pandas as pd
st.title('FineTuning SLM')
#app info
# st.markdown("###### FineTuning SLM")
# st.markdown("###### Your Text")

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
    article = st.text_input("Type your article here")

    if article :
        print("generting output ")
        info_text.text("generating output")
        output = run_fine_tuned_model(fine_tuned_model, article)
        info_text.text("generation done")
        # print(response['answer'])
        st.header("Output  : ")
        st.write(output)


def evaluate():
    info_text.text("evaluating models")
    results = evaluate_all_models_summary(fine_tuned_model, tokenizer)
    pandas_dataframe = pd.DataFrame(results)
    return pandas_dataframe
    # st.table(pandas_dataframe)
    print("clicked ")

if st.button("Evaluate test data"):
    dataframe = evaluate()
    st.table(dataframe)
    info_text.text("evaluation done")
