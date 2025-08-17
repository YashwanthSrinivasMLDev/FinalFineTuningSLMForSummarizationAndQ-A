import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# test_set_summarization = [
#     {'article': "This communication was printed, published, or produced and disseminated at U.S. taxpayer expense.  The contents of this document do not have the force and effect of law and are not meant to bind the public in any way, unless specifically incorporated into a contract. This document is intended only to provide clarity to the public regarding existing requirements under the law.  FAQS ABOUT AFFORDABLE CARE ACT IMPLEMENTATION PART 53 April 19, 2022 Set out below are Frequently Asked Questions (FAQs) regarding implementation of certain provisions of the Affordable Care Act (ACA). These FAQs have been prepared jointly by the Departments of Labor, Health and Human Services (HHS), and the Treasury (collectively, the Departments). Like previously issued FAQs (available at https://www.dol.gov/agencies/ebsa/about-ebsa/our-activities/resourcecenter/faqs and http://www.cms.gov/cciio/resources/fact-sheets-and-faqs/index.html), these FAQs answer questions from stakeholders to help people understand the law and promote compliance. Transparency in Coverage Machine-Readable Files  The Transparency in Coverage Final Rules (the TiC Final Rules) require non-grandfathered group health plans and health insurance issuers offering non-grandfathered coverage in the group and individual markets to disclose, on a public website, information regarding in-network rates for covered items and services, out-of-network allowed amounts and billed charges for covered items and services, and negotiated rates and historical net prices for covered prescription drugs in three separate machine-readable files.1 The machine-readable file requirements of the TiC Final Rules are applicable for plan years (in the individual market, policy years) beginning on or after January 1, 2022. The Departments previously announced that they will defer enforcement of the requirements related to machine-readable files disclosing in-network and out-of-network data until July 1, 2022.2 The Departments also previously announced that they will defer enforcement of the requirement that plans and issuers publish a machine-readable file related to prescription drugs while they consider, through notice-and-comment rulemaking, whether this requirement remains appropriate.3   The TiC Final Rules require plans and issuers to publish all applicable rates, which may include one or more of the following: negotiated rates, underlying fee schedule rates, or derived amounts for all covered items and services in the In-network Rate File. The Departments specify in the preamble to the TiC Final Rules that the In-network Rate File requirement applies to plans and issuers regardless of the type of payment model or models under which they provide coverage.4 1 26 CFR 54.9815-2715A3; 29 CFR 20092590.715-2715A3; and 45 CFR 147.212; 85 FR 72158 (Nov. 12, 2020)." ,
#      'summary' : "The contents of this document do not have the force and effect of law and are not meant to bind the public in any way . This document is intended only to provide clarity to the public . This communication was printed, published, or produced and disseminated at U.S. taxpayer expense ."
#      }]

test_set_summarization = [ {'article' : """Issuers generally are not permitted under federal law and regulations to reduce premiums that are otherwise due. However, in light of the urgent need to help individuals and small employers experiencing economic hardship maintain continuous coverage through the COVID-19 public health emergency, CMS will adopt a policy of relaxed enforcement with respect to 45 CFR 147.102, 155.200(f)(4), 155.400(e) and (g), 155.706(b)(6)(1)(A), 156.80(d), 156.210(a), and 156.286(a)(2)–(4) to allow issuers, on a temporary basis, to offer premium credits for 2020 coverage in the manner outlined in this bulletin. CMS encourages states to adopt a similar approach and, under this temporary exercise of enforcement discretion, will also not consider a state to have failed to substantially enforce applicable federal requirements under the Public Health Service Act (PHS Act) or the Patient Protection and Affordable Care Act (PPACA) if the state permits issuers to provide premium credits in the manner outlined in this bulletin. Similarly, in states where CMS is the primary enforcer of the applicable federal requirements, CMS will adopt a policy of relaxed enforcement to temporarily allow issuers to offer premium credits in the manner outlined in this bulletin.
Issuers wishing to provide premium credits for 2020 coverage must, in advance of providing these credits, receive the applicable regulator’s permission to provide premium credits as outlined in this bulletin, or CMS’s permission in states where CMS is the primary enforcer of the applicable federal requirements.2 In addition to approval from the applicable insurance regulator, an issuer seeking to provide premium credits must also receive permission from any applicable Exchange through which they offer qualified health plan (QHP) coverage, as outlined in the information reporting requirements described below.
In their requests to provide these credits, issuers must indicate the fixed percentage by which they intend to provide credits against 2020 premium amounts and the month(s) in 2020 to which the credits would apply. This fixed percentage credit must be offered uniformly in a non-discriminatory manner3 to all members in a market in a state without regard to whether the plan is offered through or outside an Exchange, or whether the member is eligible for advance payments of the premium tax credit (APTC).
""",
'summary' : """Normally, federal law prohibits issuers from reducing health insurance premiums. However, due to COVID-19 hardships, CMS is temporarily relaxing enforcement of certain regulations to allow issuers to offer premium credits for 2020 coverage. States are encouraged to adopt the same approach, and CMS will not penalize them for doing so.
Issuers who wish to provide premium credits must:
Obtain approval from their state regulator (or CMS in states where CMS enforces federal rules).
Get permission from any Exchanges where their Qualified Health Plans (QHPs) are offered.
Specify in advance the fixed percentage credit and applicable months.
Apply the credit uniformly and non-discriminatorily across all members in a state market, regardless of Exchange participation or tax credit eligibility.
👉 Essentially, this is a temporary COVID-19 policy allowing insurers to give premium discounts—something normally restricted—if they follow CMS and state approval processes.
"""}]

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
    # results.append(evaluate_model_summary(model=None,tokenizer=None,model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    # results.append(evaluate_model_summary(model=None,tokenizer=None,model_name="microsoft/phi-2"))
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
        prompt = f""" You are a helpful AI assistant who summarizes articles. Summarize this: {sample['article']} """
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