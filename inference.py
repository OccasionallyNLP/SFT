import torch
from accelerate import Accelerator
from accelerate import PartialState
from accelerate.utils import gather_object
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TextStreamer, BitsAndBytesConfig
from utils.utils import *
from utils.metrics import *
import pickle
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--use_flash_attention_2', action='store_true')
    parser.add_argument('--add_bos_token', action='store_true')
    args = parser.parse_args()
    return args


def get_tokenizer_and_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.use_flash_attention_2:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map={"": accelerator.process_index}, attn_implementation="flash_attention_2", trust_remote_code=True, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side='left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

# QA
def get_prompt(tokenizer, title, context, question, tokenize=False, add_bos_token=False):
    document = title+' '+context
    prompt = f"질문:{question}\n지문:{document}\n답변:"
    if add_bos_token:
        prompt = tokenizer.bos_token + prompt
    return prompt

if __name__ == "__main__":
    seed_everything(42)
    args = get_args()
    os.makedirs(args.output_dir, exist_ok = True)
    accelerator = Accelerator()
    accelerator.wait_for_everyone()    

    # tokenizer, model
    tokenizer, model  = get_tokenizer_and_model(args)
    
    test_data = load_jsonl(args.test_data)
    batch_size = args.batch_size
    
    # We set it to 8 since it is better for some hardware. 
    pad_to_multiple_of = 8
    padding_side_default = tokenizer.padding_side
    tokenizer.padding_side = "left"
    for i in test_data:
        prompt = get_prompt(tokenizer, i['title'],i['context'], i['question'], tokenize=False, add_bos_token=args.add_bos_token)
        i['input']=prompt
    formatted_prompts = [[j['input'] for j in test_data[i : i + batch_size]] for i in range(0, len(test_data), batch_size)]
    tokenized_prompts = [
    tokenizer(formatted_prompt, padding=True, pad_to_multiple_of=pad_to_multiple_of, add_special_tokens=False, return_tensors="pt")
    for formatted_prompt in formatted_prompts
    ]
    tokenizer.padding_side = padding_side_default
    terminators = [tokenizer.eos_token_id]
    streamer = None
    if accelerator.is_main_process and batch_size == 1:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) 
    completions_per_process = []
    with accelerator.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
        for batch in tqdm(batched_prompts,disable=accelerator.is_main_process != True):
            batch = batch.to(accelerator.device)
            length = batch['input_ids'].size(1)
            print(length)
            # just greedy
            outputs = model.generate(input_ids = batch['input_ids'], streamer=streamer, eos_token_id=terminators, pad_token_id = tokenizer.eos_token_id, num_beams=1, max_new_tokens=args.max_new_tokens, do_sample=False)
            outputs = outputs[:,length:].contiguous()
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            accelerator.wait_for_everyone()   
            completions_per_process.extend(generated_text)
            
    completions_gather = gather_object(completions_per_process)
    completions = completions_gather[: len(test_data)]
    accelerator.wait_for_everyone()   
    if accelerator.is_main_process:
        for i,j in zip(test_data, completions):
            i['predict']=j
        save_jsonl(args.output_dir, test_data, 'attached')
        
        actuals = [[i['answer']] for i in test_data]
        scores = get_scores(completions, actuals)
        print(scores)
        with open(os.path.join(args.output_dir,'score.txt'), 'w') as f:
            f.write(str(scores))
        with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
