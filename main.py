from TrainingSLM import load_insurace_datasets, create_model, train_model, \
    load_training_data_for_multi_task_fine_tuning, test_model_after_fine_tuning
import torch
import traceback
from create_training_data_for_instruction_finetuning import create_training_data_summary_data_for_if

if torch.cuda.is_available():
    torch.cuda.empty_cache()



EPOCH_CONTINUED_TRAINING = 2
EPOCH_FINE_TUNING = 3
ACCUMULATION_STEPS_CONTINUED_TRAINING = 15
ACCUMULATION_STEPS_FINETUNING = 15


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print('device ', device )

try :
    train_dataloader_continued_training = load_insurace_datasets()
    print("dataloader : ", train_dataloader_continued_training)

    model_tiny_llama = create_model()
    print("model : ", model_tiny_llama )

    #doing continued training on insurance datasets
    print('continued training on  insurance datasets')
    model_trained_CT_on_insurance_datasets_tiny_llama = train_model(model_tiny_llama,
                                                                    train_dataloader_continued_training,
                                                                    epochs = EPOCH_CONTINUED_TRAINING,
                                                                    accumulation_steps= ACCUMULATION_STEPS_CONTINUED_TRAINING)

    model_trained_CT_on_insurance_datasets_tiny_llama.to("cpu")
    torch.cuda.empty_cache()
    #creating summary training data
    training_data_summaries = create_training_data_summary_data_for_if()
    #creating dataloader for summary training data
    train_dataloader_summary_training_data = load_training_data_for_multi_task_fine_tuning(training_data_summaries)
    #fiinetuning  the model on summary training data
    print('finetuning on summary training data')
    model_trained_CT_on_insurance_datasets_tiny_llama.to("cuda")
    model_trained_FT_on_summary_tiny_llama = train_model(model_trained_CT_on_insurance_datasets_tiny_llama,
                                                         train_dataloader_summary_training_data,
                                                         epochs =EPOCH_FINE_TUNING ,
                                                         accumulation_steps= ACCUMULATION_STEPS_FINETUNING)


    print("saving model to disk")
    # torch.save(model_trained_FT_on_summary_tiny_llama.state_dict(), "./artifiacts/model_trained_FT_on_summary_tiny_llama")


    print("generting output ")
    output = test_model_after_fine_tuning(model_trained_FT_on_summary_tiny_llama, "This interim final rule with comment period (IFC) revises regulations to strengthen CMS’ ability to enforce compliance with Medicare and Medicaid long-term care (LTC) facility requirements for reporting information related to coronavirus disease 2019 (COVID-19), establishes a new requirement for LTC facilities for COVID-19 testing of facility residents and staff, establishes new requirements in the hospital and critical access hospital (CAH) Conditions of Participation (CoPs) for tracking the incidence and impact of COVID-19 to assist public health officials in detecting outbreaks and saving lives, and establishes requirements for all CLIA laboratories to report COVID-19 test results to the Secretary of Health and Human Services (Secretary)in such form and manner, and at such timing and frequency, as the Secretary may prescribe during the Public Health Emergency (PHE). This IFC updates the extraordinary circumstances exceptions granted for the ESRD Quality Incentive Program (QIP), Hospital Acquired Condition (HAC) Reduction Program, Hospital Readmissions Reduction Program")
    print("output of finetuned model : ", output)
    print("all done ")

except Exception :
    print("exception : ",traceback.print_exc() )
    print("cuda memory : ",torch.cuda.memory_summary())


