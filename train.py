# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
from tqdm import tqdm
import logging
import time
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from transformers import TextStreamer
from accelerate import Accelerator
from accelerate.utils import DummyOptim, DummyScheduler
import argparse
from utils.data_utils import *
from utils.utils import *
from utils.metrics import *

def get_args():
    # parser
    parser = argparse.ArgumentParser()
    # input data
    group = parser.add_argument_group(title = 'input data')
    group.add_argument('--train_data', type=str, help = 'train_data 위치',required=True)
    group.add_argument('--val_data', type=str, help='val data 위치')
    
    # logging 관련
    group = parser.add_argument_group(title = 'logs')
    group.add_argument('--logging_term', type=int, default=100)
   
    # output 
    group = parser.add_argument_group(title = 'output')
    group.add_argument('--output_dir', type=str, required=True)
   
    # 학습 관련
    group = parser.add_argument_group(title = 'train arguments')
    group.add_argument('--epochs', type=int, default = 5)
    group.add_argument('--eval_epoch', type = int, default = 1, help = 'term of evaluation')
    group.add_argument('--batch_size', default = 8, type=int)
    group.add_argument('--eval_batch_size', default = 8, type=int)
    group.add_argument('--lr', type=float, default = 5e-5)
    group.add_argument('--gradient_clipping', type=float, default = 1.)
    group.add_argument('--warmup', type=float, default = 0.)
    group.add_argument('--decay', type=float, default = 0.1)
    group.add_argument('--accumulation_steps', type=int, default = 1) 
    
    # PTM model
    group = parser.add_argument_group(title = 'model')
    group.add_argument('--plm_path', type=str)
    group.add_argument('--use_flash_attention_2', action='store_true')
    
    # model input
    group = parser.add_argument_group(title = 'model_input')
    group.add_argument('--max_length', type=int)
    group.add_argument('--eval_max_length', type=int)
    group.add_argument('--add_bos_token', action='store_true')
    group.add_argument('--add_eos_token', action='store_true')
    group.add_argument('--task', type=str)
    
    # generation 
    group = parser.add_argument_group(title = 'generation')
    group.add_argument('--generate', action='store_true')
    group.add_argument('--max_new_tokens', type=int)
    
    # early stop 관련
    group = parser.add_argument_group(title = 'early_stop')
    group.add_argument('--early_stop', action='store_true')
    group.add_argument('--early_stop_metric', type=str, default = 'loss')
    group.add_argument('--early_stop_metric_is_max_better', action='store_true')
    group.add_argument('--patience', type=int, default = 3)
    group.add_argument('--save_model_every_epoch', action='store_true')
    
    args  = parser.parse_args()
    return args

def get_tokenizer_and_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path,trust_remote_code=True)
    # for gpt 4 model 
    if tokenizer.eos_token is None:
        tokenizer.eos_token = '<|endoftext|>'
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    
    if args.use_flash_attention_2:
        model = AutoModelForCausalLM.from_pretrained(args.plm_path, attn_implementation="flash_attention_2", trust_remote_code=True, torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.plm_path, trust_remote_code=True)
    return tokenizer, model 

def load_dataloaders(args, tokenizer):
    # LOAD DATASETS
    train_data = load_jsonl(args.train_data)
    if args.val_data is None:
        val_data = random.sample(train_data, k=len(train_data)*0.01)
    else:
        val_data = load_jsonl(args.val_data)
    
    #### datasets
    if args.task == 'qa':
        pass
    elif args.task == 'nli':
        train_dataset = NLIDataset(train_data, tokenizer, args.max_length, args.eval_max_length, args.add_bos_token, args.add_eos_token)
        val_dataset = NLIDataset(val_data, tokenizer, args.max_length, args.eval_max_length, args.add_bos_token, args.add_eos_token)
    elif args.task == 'ynat':
        pass
    
    # sampler
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, collate_fn = train_dataset.collate_fn, sampler = train_sampler)
    val_dataloader = DataLoader(val_dataset,batch_size = args.eval_batch_size, collate_fn = val_dataset.collate_fn)
    return train_dataset, train_dataloader, val_dataset, val_dataloader


def train():
    if accelerator.is_main_process:
        early_stop = EarlyStopping(args.patience, args.output_dir, max = args.early_stop_metric_is_max_better, min_difference=1e-5, model_save_dict=False)
    flag_tensor = torch.zeros(1).cuda()
    ########################################################################################
    # train
    ########################################################################################
    global_step = 0
    optimizer.zero_grad()            
    for epoch in range(1, args.epochs+1):
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0.
        step = 0
        iter_bar = tqdm(train_dataloader, desc='step', disable=not accelerator.is_main_process)
        for data in iter_bar:
            step+=1
            data = {i:j.to(accelerator.device) for i,j in data.items()}
            out = model.forward(\
                                input_ids = data['input_ids'], \
                                attention_mask = data['attention_mask'], \
                                labels = data['labels'])
            loss = out.loss
            loss = loss / args.accumulation_steps
            accelerator.backward(loss)
            if step%args.accumulation_steps==0 or (
                len(train_dataloader) <= args.accumulation_steps
                and (step) == len(train_dataloader)
        ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step+=1
            epoch_loss+=loss.mean().item()*args.accumulation_steps
            iter_bar.set_postfix({'epoch':epoch, 'global_step':global_step, 'lr':f"{scheduler.get_last_lr()[0]:.5f}",'epoch_loss':f'{epoch_loss/step:.5f}'}) 
            if args.logging_term is not None:
                if global_step%args.logging_term == 0:
                    if accelerator.is_main_process:
                        logger1.info(iter_bar)
                        logger2.info(iter_bar)
            
        # epoch 당 기록.
        if accelerator.is_main_process:
            logger1.info(iter_bar)
            logger2.info(iter_bar)
        ########################################################################################
        # evaluation
        ###################################################################################################
        if args.eval_epoch!=0 and epoch%args.eval_epoch==0:
            val_score, predicts = evaluation(args, accelerator, model, tokenizer, val_dataloader, args.generate)
            torch.cuda.empty_cache()
            
            if accelerator.is_main_process:
                if args.generate:
                    actuals = [val_dataset.get_example(i) for i in range(len(predicts))]
                    # XXX
                    # get your label. NLI
                    actuals = [[STR2LABEL[i]] for i in actuals]
                    scores = get_scores(predicts, actuals)
                else:
                    scores = {}
                scores['loss']=val_score
                
                logger1.info(f'Val ---- epoch : {epoch} ----- scores:{scores}')
                logger2.info(f'Val ---- epoch : {epoch} ----- scores:{scores}')
#                 wandb.info(scores)
                upwrapped_model = accelerator.unwrap_model(model)
                if args.save_model_every_epoch:
                    save_path = os.path.join(args.output_dir,'model_%d'%epoch)
                    os.makedirs(save_path, exist_ok = True)
                    upwrapped_model.save_pretrained(save_path, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
                    tokenizer.save_pretrained(save_path)
                early_stop.check(upwrapped_model, scores[args.early_stop_metric])  
                if early_stop.timetobreak:
                    flag_tensor += 1
        accelerator.wait_for_everyone()
        #############################################################################r######################
        if args.early_stop:   
            to_stop = sum(accelerator.gather(flag_tensor)).item()
            if to_stop>=1:
                if accelerator.is_main_process:
                    logger1.info('early stop')
                    logger2.info('early stop')
                    save_path = os.path.join(args.output_dir,'best_model')
                    os.makedirs(save_path, exist_ok = True)
                    early_stop.best_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger1.info('train_end')
                    logger2.info('train end')
                break


# evaluation
def evaluation(args, accelerator, model, tokenizer, eval_dataloader, generate=True):
    total_loss = 0.
    model.eval()
    predicts = []
    actuals = []
    losses = []
    with torch.no_grad():
        for data in tqdm(eval_dataloader, desc = 'evaluate', disable =  not accelerator.is_main_process):
            data = {i:j.to(accelerator.device) for i,j in data.items()}
            output = model.forward(\
                                   input_ids = data['input_ids'],\
                                   attention_mask = data['attention_mask'],\
                                   labels = data['labels'])
            loss = output.loss
            total_loss+=loss
            
            if generate:
                model_to_generate = model.module if hasattr(model,'module') else model
                max_new_tokens = args.max_new_tokens
                streamer=None
                if args.eval_batch_size == 1:
                    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                predict = model_to_generate.generate(
                    input_ids = data['eval_input_ids'],
                    pad_token_id = tokenizer.pad_token_id,
                    do_sample = False,
                    num_beams = 1,
                    max_new_tokens=args.max_new_tokens,
                    streamer = streamer)
                
                # predict postprossessing
                input_len = data['eval_input_ids'].size(1) # bs, seq_len
                predict = predict[:,input_len:].contiguous()
                
                if accelerator.distributed_type!='NO':
                    padded_predict = accelerator.pad_across_processes(predict, dim=1, pad_index=tokenizer.pad_token_id, pad_first=False)
                    predict = accelerator.gather_for_metrics(padded_predict)
                    
                predict = tokenizer.batch_decode(predict, skip_special_tokens=True)
                predicts.extend(predict)
                
        total_loss = total_loss/len(eval_dataloader)
        if accelerator.distributed_type!='NO':
            losses = accelerator.gather(total_loss)
        else:
            losses = torch.tensor([total_loss])
    
    score = losses.sum().item()/len(losses)
    return score, predicts

if __name__=='__main__':
    # prepare
    args  = get_args()
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok = True)
    logger1, logger2 = get_log(args)
    
    #args.local_rank = int(os.environ["LOCAL_RANK"])
    
    ########################################################################################
    # tokenizer, model load
    ########################################################################################
    tokenizer, model = get_tokenizer_and_model(args)
    ########################################################################################

    ########################################################################################
    # accelerator
    ########################################################################################
    accelerator = Accelerator()
    if accelerator.state.deepspeed_plugin is not None:
        print(accelerator.state.deepspeed_plugin.deepspeed_config)
    model.to(accelerator.device)
    if accelerator.is_main_process:
        logger1.info(args)
        logger2.info(args)
    # save
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir,'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
#         wandb.init(project='ACL_2025')
#         wandb.run.name = args.test_name
#         wandb.run.save()
#         wandb.config.update(args)
    ########################################################################################

    ########################################################################################
    # data
    ########################################################################################
    train_dataset, train_dataloader, val_dataset, val_dataloader = load_dataloaders(args, tokenizer)
    ########################################################################################

    ########################################################################################
    # optimizer, scheduler, synchronize
    ########################################################################################
    optimizer_grouped_parameters = make_optimizer_group(model, args.decay)
    # New Code #
    # Creates Dummy Optimizer if `optimizer` was spcified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.decay)
    
    # Get gradient accumulation steps from deepspeed config if available
    if accelerator.state.deepspeed_plugin is not None:
        args.gradient_accumulation_steps = accelerator.state.deepspeed_plugin.deepspeed_config[
            "gradient_accumulation_steps"
        ]
    # Scheduler and math around the number of training steps.
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # else:
    #     args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # New Code #
    # Creates Dummy Scheduler if `scheduler` was spcified in the config file else creates `args.lr_scheduler_type` Scheduler
    t_total = len(train_dataloader)*args.epochs//args.accumulation_steps
    n_warmup = int(t_total*args.warmup) if args.warmup<1 else int(args.warmup)//(torch.cuda.device_count())
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=n_warmup, num_training_steps=t_total)
    else:
        scheduler = DummyScheduler(optimizer, total_num_steps=t_total, warmup_num_steps=n_warmup)
    ########################################################################################
    ########################################################################################
    # prepare
    ########################################################################################
    model, optimizer, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    ########################################################################################
    ########################################################################################
    # train
    ########################################################################################
    train()
    print('done')
    ########################################################################################
