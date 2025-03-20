from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import torch
from rich import print
import math
from grpo import GRPO


SYSTEM_PROMPT = "Respond in following format:<thinking>{step by step reasoning}</thinking><answer>{number}</answer>"


def prepare_dataset(dataset) -> Dataset:
    extract_hash_answer = (
        lambda text: text.split("####")[1].strip() if "####" in text else None
    )

    def process_example(example: dict):
        answer = extract_hash_answer(example["answer"])
        if answer is None:
            return None
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
            "answer": answer,
        }

    dataset = dataset.map(
        process_example,
        remove_columns=[
            col for col in dataset.column_names if col not in ["prompt", "answer"]
        ],
    )
    dataset = dataset.filter(lambda x: x is not None)

    return dataset


# model_name = "Qwen/Qwen2.5-0.5B-Instruct"
# small models are kind of dumb, they need a little push so using this fine-tuned model
# source: https://github.com/joey00072/nanoGRPO/blob/master/cold_start/cold_start_finetune.py
# you can totally use the base model, it will just take longer to converge
# model_name = "joey00072/Llama-3.2-1B-Instruct-cold-start-ft2"

# model_name = "unsloth/Llama-3.2-1B-Instruct"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    # lora_dropout=0.1,
)
model = get_peft_model(model, lora_config)
model = model.to(torch.bfloat16)


def reward_func_len(sample: dict, s: str, *args, **kwargs):
    return 4 - (len(s)/1000)

def response_format_reward(sample: dict, s: str, *args, **kwargs):
    # print(sample.keys())
    correct_template =0
    try:
        s = s.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[1]
    except:
        return -1
    if "<|eot_id|>" in s:
        s = s.split("<|eot_id|>")[0]
    try:
        print("-"*100)
        print(s)
        print("-"*100)
    except:
        ...
    total_reward = 0
    for tag in ["<thinking>", "</thinking>", "<answer>", "</answer>"]:
        if tag in s:
            total_reward+=0.15
            if s.count(tag)>1:
                total_reward -= s.count(tag)*0.01

    if s.count("<thinking>")==1:
        total_reward += .5
    else:
        total_reward -= .1

    if s.count("</thinking><answer>")==1:
        total_reward += 1
        correct_template += 1
    else:
        if s.count("<thinking>")==1:
            total_reward += .2
        else:
            total_reward -= .1
        if s.count("<answer>")==1:
            total_reward += .2
        else:
            total_reward -= .1

    if s.count("</answer>")==1 and s.split("</answer>")[1].strip() == "":
            total_reward += 1
    else:
        total_reward -= 0.1

    if s.count("<answer>")==1:
        total_reward += .2

        r = s.split("<answer>")[1].strip()
        if "</answer>" in r:
            total_reward += .2
            if r.count("</answer>")==1:
                total_reward += 2 
                split = r.split("</answer>")
                r = split[0].strip()
                try:
                    r = float(r)
                    total_reward += 1
                    if r == float(sample["answer"]):
                        total_reward += 2
                        correct_template += 1
                except:
                    total_reward -= 0.1

                if len(split) > 1:
                    if split[1].strip() != "":
                        total_reward += 3
                        correct_template += 1
                    else:
                        total_reward -= len(split[1].strip())/1000
                else:
                    total_reward -= 0.2
            else:
                total_reward -= 0.1
        else:
            total_reward -=0.1
    if correct_template == 3:
        total_reward += 2
    return total_reward



# dataset = load_dataset("openai/gsm8k", "main")["train"]
# dataset = prepare_dataset(dataset)
from data import PicoThinkingFunctionCalling

dataset = PicoThinkingFunctionCalling()

group_size = 8
micro_group_size =2
lr = 5e-5
weight_decay = 0.1
reward_functions = [
    response_format_reward,
]
beta = 0.01
print(model)

ref_model = None
trainer = GRPO(
    model,
    ref_model,
    tokenizer=tokenizer,
    group_size=group_size,
    micro_group_size=micro_group_size,
    dataset=dataset,
    reward_functions=reward_functions,
    log_wandb=True,
    lr=lr,
    weight_decay=weight_decay,
    beta=beta,
    dtype=torch.bfloat16
)

trainer.train()
