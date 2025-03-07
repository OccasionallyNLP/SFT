import torch
from accelerate import PartialState
from accelerate.utils import gather
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import classification_report
import os
LABEL_SWITCH = {0: 2, 1: 1, 2: 1}
ORG_LABLE_SWITCH = {2: 0, 1: 1, 1: 2}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--use_flash_attention_2', action='store_true')
    args = parser.parse_args()
    return args


def get_tokenizer_and_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # for gpt 4 model
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"  # for classification
    
    if args.use_flash_attention_2:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=distributed_state.device
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path, trust_remote_code=True, device_map=distributed_state.device)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model

if __name__ == "__main__":
    args = get_args()
    # sanity check
    os.makedirs(args.output_dir, exist_ok = True)
    distributed_state = PartialState()

    # tokenizer, model
    tokenizer, model  = get_tokenizer_and_model(args)
    
    ##########################################################################
    # data
    ##########################################################################
    test_data = load_dataset(args.data_name, split=args.split).select(indices=range(128))
    batch_size = args.batch_size
    _test_data = test_data.to_list()
    # We set it to 8 since it is better for some hardware. 
    pad_to_multiple_of = 8
    padding_side_default = tokenizer.padding_side
    for i in _test_data:
        premise = i['premise']
        hypothesis = i['hypothesis']
        label = i['label']
        prompt = "Premise: " + premise + "\n\nHypothesis: " + hypothesis
        i['input'] = prompt
    formatted_prompts = [[j['input'] for j in _test_data[i : i + batch_size]] for i in range(0, len(_test_data), batch_size)]
    tokenized_prompts = [
    tokenizer(formatted_prompt, padding=True, pad_to_multiple_of=pad_to_multiple_of, add_special_tokens=False, return_tensors="pt") for formatted_prompt in formatted_prompts
    ]
    
    # evaluate
    predicts = []
    with distributed_state.split_between_processes(tokenized_prompts, apply_padding=True) as batched_prompts:
        
        for data in tqdm(batched_prompts):
            data = data.to(distributed_state.device)
            output = model.forward(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
            )
            predicts.extend(output.logits.argmax(dim=-1))
        
        try:
            predicts = gather(
                torch.tensor(predicts, device=distributed_state.device)
            )
        except:
            pass    
    predicts = predicts.cpu().tolist()
    _predicts = [ORG_LABLE_SWITCH[i] for i in predicts]
    if distributed_state.is_main_process:
        test_data = test_data.add_column("predict", _predicts)
        test_data.to_json(os.path.join(args.output_dir, 'attached.jsonl'))
        actuals = [LABEL_SWITCH[i] for i in test_data['label']]
        report = classification_report(actuals, predicts)
        print(classification_report(actuals, predicts))