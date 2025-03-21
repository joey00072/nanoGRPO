from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
import torch
from rich import print
import math
from grpo import GRPO
import re
import json

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



def reward_func(sample: dict, response: str, *args, **kwargs):
    
    try:
        reward = 0
        _, resp = response.split("<think>")
        sp = resp.split("</think>")
        if len(sp)==2:
            thinking,answer = sp
        else:
            thinking = sp[0]
        
        OPEN_FUNC_CALL = "<function_call>"
        CLOSE_FUNC_CALL = "</function_call>"
        OPEN_REQUEST = "<request>"
        CLOSE_REQUEST = "</request>"
        OPEN_RESPONSE = "<response>"
        CLOSE_RESPONSE = "</response>"

        func_calls_tags =[OPEN_FUNC_CALL, CLOSE_FUNC_CALL, OPEN_REQUEST, CLOSE_REQUEST, OPEN_RESPONSE, CLOSE_RESPONSE]
        
        for tag in func_calls_tags:
            if tag in thinking:
                reward += 0.3
            if tag in answer:
                reward -= 0.1
                
        if thinking.count(OPEN_FUNC_CALL)>0:
            if thinking.count(OPEN_FUNC_CALL) == thinking.count(CLOSE_FUNC_CALL):
                reward += 0.3
            else:
                reward -= 0.1
            
        if answer.count(OPEN_FUNC_CALL)>0:
            if answer.count(OPEN_FUNC_CALL) == answer.count(CLOSE_FUNC_CALL):
                reward += 0.3
            else:
                reward -= 0.1
                
        if thinking.count(OPEN_REQUEST)>0:
            if thinking.count(OPEN_REQUEST) == thinking.count(CLOSE_REQUEST):
                reward += 0.3
            else:
                reward -= 0.1
                

        pattern = re.compile(
            r"""
            <function_call>\s*
            <request>\s*
            (.*?)\s*
            </request>\s*
            <response>\s*
            (.*?)\s*
            </response>\s*
            </function_call>
            """, re.VERBOSE | re.IGNORECASE | re.DOTALL
        )
        
        matches = pattern.findall(thinking)
        valid_count = 0
        
        for request_json, response_json in matches:
            try:
                json.loads(request_json.strip())
                json.loads(response_json.strip())
                valid_count += 1
            except json.JSONDecodeError:
                reward -= 0.1
            
        return reward + valid_count *2
    except Exception as e:
        print(e)
        return 0



# dataset = load_dataset("openai/gsm8k", "main")["train"]
# dataset = prepare_dataset(dataset)
from data import PicoThinkingFunctionCalling

dataset = PicoThinkingFunctionCalling()

group_size = 8
micro_group_size =2
lr = 5e-5
weight_decay = 0.1
reward_functions = [
    reward_func,
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
    dtype=torch.bfloat16,
    push_to_hub=True,
    push_checkpoint_name="joey00072/pico_thinking_function_calling_grpo",
    push_interval=16,
)

trainer.train()
