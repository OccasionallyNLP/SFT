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

    
@dataclass
class QADataset(BaseDataset):
    '''
    {'title': '제주도 장마 시작 … 중부는 이달 말부터',
 'context': '올여름 장마가 17일 제주도에서 시작됐다. 서울 등 중부지방은 예년보다 사나흘 정도 늦은 이달 말께 장마가 시작될 전망이다.17일 기상청에 따르면 제주도 남쪽 먼바다에 있는 장마전선의 영향으로 이날 제주도 산간 및 내륙지역에 호우주의보가 내려지면서 곳곳에 100㎜에 육박하는 많은 비가 내렸다. 제주의 장마는 평년보다 2~3일, 지난해보다는 하루 일찍 시작됐다. 장마는 고온다습한 북태평양 기단과 한랭 습윤한 오호츠크해 기단이 만나 형성되는 장마전선에서 내리는 비를 뜻한다.장마전선은 18일 제주도 먼 남쪽 해상으로 내려갔다가 20일께 다시 북상해 전남 남해안까지 영향을 줄 것으로 보인다. 이에 따라 20~21일 남부지방에도 예년보다 사흘 정도 장마가 일찍 찾아올 전망이다. 그러나 장마전선을 밀어올리는 북태평양 고기압 세력이 약해 서울 등 중부지방은 평년보다 사나흘가량 늦은 이달 말부터 장마가 시작될 것이라는 게 기상청의 설명이다. 장마전선은 이후 한 달가량 한반도 중남부를 오르내리며 곳곳에 비를 뿌릴 전망이다. 최근 30년간 평균치에 따르면 중부지방의 장마 시작일은 6월24~25일이었으며 장마기간은 32일, 강수일수는 17.2일이었다.기상청은 올해 장마기간의 평균 강수량이 350~400㎜로 평년과 비슷하거나 적을 것으로 내다봤다. 브라질 월드컵 한국과 러시아의 경기가 열리는 18일 오전 서울은 대체로 구름이 많이 끼지만 비는 오지 않을 것으로 예상돼 거리 응원에는 지장이 없을 전망이다.',
 'news_category': '종합',
 'source': 'hankyung',
 'guid': 'klue-mrc-v1_train_12759',
 'is_impossible': False,
 'question_type': 1,
 'question': '북태평양 기단과 오호츠크해 기단이 만나 국내에 머무르는 기간은?',
 'answers': {'answer_start': [478, 478], 'text': ['한 달가량', '한 달']}}
    '''
    def __init__(self, data, tokenizer, max_length=None, eval_max_length=None, \
                 add_bos_token=False, add_eos_token=True):
        super().__init__(data, tokenizer, max_length, eval_max_length, add_bos_token, add_eos_token)

    def __getitem__(self, index):
        document = self.data[index]['title']+' '+self.data[index]['context']
        question = self.data[index]['question']
        prompt = f"문서: {premise}\n질문: {hypothesis}\n정답: "
        if self.add_bos_token:
            prompt = self.tokenizer.bos_token+prompt
        answer = self.get_answer(self.data[index])
        if self.add_eos_token:
            answer = answer+self.tokenizer.eos_token
        prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        response = self.tokenizer.encode(answer, add_special_tokens=False)
        return dict(prompt=prompt, response=response)
    
    def get_answer(self, example):
        if 'answers' in example:
            answer = random.sample(example['answers']['text'], k=1)[0]
        return answer
