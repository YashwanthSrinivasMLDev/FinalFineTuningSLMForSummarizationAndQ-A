import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from testing_data.summarization_testing_datat import  test_set_summarization
from testing_data.qa_testing_data import test_set_qa
rouge = evaluate.load('rouge')

# models_to_test = [
#     {'model': }
# ]
def set_up_all_models():

    return True

def evaluate_all_models_summary(model_use_case, model_finetuned,tokenizer_finedtuned=None, ):
    print("inside evaluate all models, model use case " , model_use_case)
    # models =[
    #     {'model_name': "microsoft/phi-2",
    #      'already_finetuned' : 'yes',
    #      }
    # ]
    # our fine-tuned model
    results_finetuned = evaluate_model(model_use_case, model_finetuned,tokenizer_finedtuned,  model_name="finetuned_model(microsoft/phi-2)")
    # print('results_finetuned ' , results_finetuned)
    # for hugging face models

    #foundational slm
    results_fm_TinyLlama= evaluate_model(model_use_case, model=None,tokenizer=None , model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    # print('results_fm_TinyLlama ' , results_fm_TinyLlama)

    #gpt 3.5
    # results.append(evaluate_model(model=None,tokenizer=None,model_name="Xenova/gpt-3.5-turbo-16k"))
    # print(len(results_finetuned), len(results_fm_TinyLlama))
    # print(type(results_finetuned), type(results_fm_TinyLlama))
    combined_results = [ {**dict1, **dict2} for dict1, dict2 in  zip(results_finetuned, results_fm_TinyLlama)]

    # print("evaluation of all models done : ", combined_results )
    return combined_results


def  evaluate_model( model_use_case,  model=None, tokenizer=None,  model_name=None ):
    print("evaluating for model : ", model_name, " for use case : ", model_use_case)
    device= "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens = 30
    min_new_tokens = 5
    results = []
    rogue_scores =[]
    articles=[]
    ground_truth_summaries = []
    generated_summaries = []

    questions = []
    ground_truth_answers = []
    generated_answers = []


    if model is None  :
        # external models
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer= AutoTokenizer.from_pretrained(model_name)

    model.to(device)

    if model_use_case=="summarization":
        for i, sample in enumerate(test_set_summarization):
            # print(i,sample)
            articles.append(sample['article'])
            ground_truth_summaries.append(sample['summary'])
            prompt = f""" You are a helpful AI assistant.
            Provide a concise summary of the following article:
            
            Article : 
            {sample['article']} """

            #code from trainingSLM.py
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            max_context = model.config.max_position_embeddings

            # Leave space for new tokens
            max_allowed_prompt_len = max_context - max_new_tokens
            if prompt_len > max_allowed_prompt_len:
                print(f"⚠️ Prompt too long ({prompt_len} tokens). Trimming to {max_allowed_prompt_len} tokens.")
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, add_special_tokens=False,
                                   max_length=max_allowed_prompt_len).to(
                    model.device)

            with torch.no_grad() :
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.1
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_summaries.append(summary)


            result = rouge.compute(
                predictions=[summary],
                references=[sample['summary']],
                # rouge_types=['rouge1','rouge2','rougeL']
                rouge_types=['rougeL']
            )

            results.append({
                'article': sample['article'],
                'summary': sample['summary'],
                f" {model_name}_rougeL": result['rougeL'],
                f" {model_name}_predicted_answer": summary,
            })

        del model
        torch.cuda.empty_cache()
        return  results

    elif model_use_case == "qa":
        for i, sample in enumerate(test_set_qa):
            if i > 1:
                break
            # print(i,sample)
            questions.append(sample['question'])
            ground_truth_answers.append(sample['answer'])

        #    prompt = f""" You are a helpful AI assistant who
        # answers questions. Answer this question {sample['question']} """

            prompt = f""" You are a helpful AI assistant. 
                Provide a concise and accurate answer to the following question. 
            
            Question:
                {sample['question']} """

            #code from trainingSLM.py
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            max_context = model.config.max_position_embeddings

            # Leave space for new tokens
            max_allowed_prompt_len = max_context - max_new_tokens
            if prompt_len > max_allowed_prompt_len:
                print(f"⚠️ Prompt too long ({prompt_len} tokens). Trimming to {max_allowed_prompt_len} tokens.")
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, add_special_tokens=False,
                                   max_length=max_allowed_prompt_len).to(
                    model.device)

            with torch.no_grad() :
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.1
                )

            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_answers.append(answer)

            # print("article : ", articles )
            # print("ground truth summaries :", ground_truth_summaries )
            # print('generated summaries', generated_summaries)
            result =  rouge.compute(
                predictions=[answer] ,
                references= [sample['answer']],
                # rouge_types=['rouge1','rouge2','rougeL']
                rouge_types=['rougeL']
            )

            results.append({
                'question' : sample['question'],
                'answer' : sample['answer'],
                f" {model_name}_rougeL" : result['rougeL'] ,
                f" {model_name}_predicted_answer" :  answer ,
            })
        del model
        torch.cuda.empty_cache()
        return  results