import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
test_set_summarization = [
    {'article': "This communication was printed, published, or produced and disseminated at U.S. taxpayer expense.  \n \nThe contents of this document do not have the force and effect of law and are not meant to bind the public in any \nway, unless specifically incorporated into a contract. This document is intended only to provide clarity to the public \nregarding existing requirements under the law.  \n \nFAQS ABOUT AFFORDABLE CARE ACT \nIMPLEMENTATION PART 53 \n \nApril 19, 2022 \n \nSet out below are Frequently Asked Questions (FAQs) regarding implementation of certain \nprovisions of the Affordable Care Act (ACA). These FAQs have been prepared jointly by the \nDepartments of Labor, Health and Human Services (HHS), and the Treasury (collectively, the \nDepartments). Like previously issued FAQs (available at \nhttps://www.dol.gov/agencies/ebsa/about-ebsa/our-activities/resourcecenter/faqs and \nhttp://www.cms.gov/cciio/resources/fact-sheets-and-faqs/index.html), these FAQs answer \nquestions from stakeholders to help people understand the law and promote compliance. \n \nTransparency in Coverage Machine-Readable Files  \n \nThe Transparency in Coverage Final Rules (the TiC Final Rules) require non-grandfathered \ngroup health plans and health insurance issuers offering non-grandfathered coverage in the group \nand individual markets to disclose, on a public website, information regarding in-network rates \nfor covered items and services, out-of-network allowed amounts and billed charges for covered \nitems and services, and negotiated rates and historical net prices for covered prescription drugs in \nthree separate machine-readable files.\n1 The machine-readable file requirements of the TiC Final \nRules are applicable for plan years (in the individual market, policy years) beginning on or after \nJanuary 1, 2022. The Departments previously announced that they will defer enforcement of the \nrequirements related to machine-readable files disclosing in-network and out-of-network data \nuntil July 1, 2022.\n2 The Departments also previously announced that they will defer enforcement \nof the requirement that plans and issuers publish a machine-readable file related to prescription \ndrugs while they consider, through notice-and-comment rulemaking, whether this requirement \nremains appropriate.3 \n  \nThe TiC Final Rules require plans and issuers to publish all applicable rates, which may include \none or more of the following: negotiated rates, underlying fee schedule rates, or derived amounts \nfor all covered items and services in the In-network Rate File. The Departments specify in the \npreamble to the TiC Final Rules that the In-network Rate File requirement applies to plans and \nissuers regardless of the type of payment model or models under which they provide coverage.\n4 \n                                               \n1 26 CFR 54.9815-2715A3; 29 CFR\u20092590.715-2715A3; and 45 CFR 147.212; 85 FR 72158 (Nov. 12, 2020)." ,
     'summary' : "The contents of this document do not have the force and effect of law and are not meant to bind the public in any way . This document is intended only to provide clarity to the public . This communication was printed, published, or produced and disseminated at U.S. taxpayer expense ."
     }]

rouge = evaluate.load('rouge')

# models_to_test = [
#     {'model': }
# ]
def set_up_all_models():

    return True

def evaluate_all_models_summary(model_finetuned,tokenizer_finedtuned=None):
    results = []
    models =[
        {'model_name': "microsoft/phi-2",
         'already_finetuned' : 'yes',
         }
    ]
    # our fine-tuned model
    results.append(evaluate_model_summary(model_finetuned,tokenizer_finedtuned, model_name="finetuned_model(microsoft/phi-2)"))

    # for hugging face models
    #foundational slm
    results.append(evaluate_model_summary(model=None,tokenizer=None,model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    #gpt 3.5
    # results.append(evaluate_model_summary(model=None,tokenizer=None,model_name="Xenova/gpt-3.5-turbo-16k"))
    print("evaluation of all models done : ", results )
    return results


def  evaluate_model_summary(  model=None, tokenizer=None,  model_name=None):
    print("evaluating for model : ", model_name)
    device= "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens = 70
    min_new_tokens = 50
    results = []
    rogue_scores =[]
    articles=[]
    ground_truth_summaries = []
    generated_summaries = []
    if model is None  :
        # external models
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer= AutoTokenizer.from_pretrained(model_name)

    model.to(device)

    for i, sample in enumerate(test_set_summarization):
        # print(i,sample)
        articles.append(sample['article'])
        ground_truth_summaries.append(sample['summary'])
        prompt = f""" You are a helpful AI assistant who summarizes 
        articles. Summarize the following article: {sample['article']} """
        # input_tokenized = tokenizer(prompt, truncation=True, return_tensors="pt").to(device)

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

    print("article : ", articles )
    print("ground truth summaries :", ground_truth_summaries )
    print('generated summaries', generated_summaries)
    result =  rouge.compute(
        predictions=generated_summaries,
        references= ground_truth_summaries,
        rouge_types=['rouge1','rouge2','rougeL']
    )

    results.append(result)
    # print('rouge results : ', results)
    # for i  in results:
    #     rogue_scores.append(i['rougeL'])
    del model
    torch.cuda.empty_cache()
    return {
            'model_name' : model_name,
            'results' : results,
        'generated_summaries' : generated_summaries
        }