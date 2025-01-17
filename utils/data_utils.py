import json
import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
import random
import copy
from dataclasses import dataclass
from transformers import AutoTokenizer
from itertools import combinations
from collections import defaultdict

OPTIONS='함의 중립 모순'
LABEL2STR={0:'함의',1:'중립',2:'모순'}

@dataclass
class BaseDataset(Dataset):
    data:List[dict]
    tokenizer:AutoTokenizer
    max_length:Optional[int]=None
    eval_max_length:Optional[int]=None
    add_bos_token:bool=False
    add_eos_token:bool=True

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def get_answer(self, example):
        raise NotImplementedError

    def get_example(self, index):
        return self.data[index]

    def get_label_mask(self, prompt:list, label:list, max_length:int)->list:
        output = [0 for _ in range(len(prompt+label))]
        for i in range(len(prompt)):
            output[i]=1
        output = output[:max_length]
        if len(output)<max_length:
            output = output+[0]*(max_length-len(output))
        return output

    def collate_fn(self, batch):
        # pad right
        max_length = self.max_length
        eval_max_length = self.eval_max_length
        if self.max_length is None:
            max_length = -1
            for b in batch:
                input_len = len(b['prompt']+b['response'])
                if input_len>max_length:
                    max_length = input_len
        if eval_max_length is None:
            eval_max_length = -1
            for b in batch:
                input_len = len(b['prompt'])
                if input_len>eval_max_length:
                    eval_max_length = input_len
        input_ids = []
        attention_mask = []
        labels = []
        eval_input_ids = []
        for b in batch:
            input_id = b['prompt']+b['response']
            length = len(input_id)
            input_id = input_id[:max_length]
            if length<max_length:
                input_id = input_id + [self.tokenizer.pad_token_id]*max(0,max_length - length)
            input_ids.append(input_id)

            attention_mask_i = [1]*length
            attention_mask_i = attention_mask_i[:max_length]
            if length<max_length:
                attention_mask_i = attention_mask_i+[0]*max(0,max_length - length)
            attention_mask.append(attention_mask_i)
            label_mask = self.get_label_mask(b['prompt'],b['response'], max_length)
            label = torch.tensor(input_id).masked_fill\
                          (torch.tensor(label_mask).bool(), self.tokenizer.pad_token_id).tolist()
            labels.append(label)
            assert len(label_mask)==len(input_id)==len(label)

            # padding_left
            eval_length = len(b['prompt'])
            eval_input = b['prompt']
            eval_input_id = [self.tokenizer.pad_token_id]*max(0, eval_max_length - eval_length)+eval_input[:eval_max_length]
            eval_input_ids.append(eval_input_id)
        return dict(input_ids = torch.tensor(input_ids), attention_mask = torch.tensor(attention_mask), labels=torch.tensor(labels), eval_input_ids = torch.tensor(eval_input_ids))  

@dataclass
class NLIDataset(BaseDataset):
    def __init__(self, data, tokenizer, max_length=None, eval_max_length=None, \
                 add_bos_token=False, add_eos_token=True):
        super().__init__(data, tokenizer, max_length, eval_max_length, add_bos_token, add_eos_token)

    def __getitem__(self, index):
        premise = self.data[index]['premise']
        hypothesis = self.data[index]['hypothesis']
        prompt = f"전제: {premise}\n가설: {hypothesis}\n주어진 가설과 전제는 어떠한 관계인가\n{OPTIONS}"
        if self.add_bos_token:
            prompt = self.tokenizer.bos_token+prompt
        answer = self.get_answer(self.data[index])
        if self.add_eos_token:
            answer = answer+self.tokenizer.eos_token
        prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        response = self.tokenizer.encode(answer, add_special_tokens=False)
        return dict(prompt=prompt, response=response)
    
    def get_answer(self, example):
        if 'label' in example:
            answer = LABEL2STR[example['label']]
        return answer

class QADataset(BaseDataset):
    def __init__(self, data, tokenizer, max_length=None, eval_max_length=None, \
                 add_bos_token=False, add_eos_token=True):
        super().__init__(data, tokenizer, max_length, eval_max_length, add_bos_token, add_eos_token)
    
    def __getitem__(self, index):
        title = self.data[index]['title']
        context = self.data[index]['context']
        document = title+' '+context
        question = self.data[index]['question']
        answer = self.data[index]['answer']
        prompt = f"질문:{question}\n지문:{document}\n답변:"
        
        if self.add_bos_token:
            prompt = self.tokenizer.bos_token+prompt
        if self.add_eos_token:
            answer = answer+self.tokenizer.eos_token
        prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        response = self.tokenizer.encode(answer, add_special_tokens=False)
        return dict(prompt=prompt, response=response)
    
    def get_answer(self, example):
        if 'answer' in example:
            answer = example['answer']
        return answer
