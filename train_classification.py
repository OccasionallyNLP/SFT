# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm
import logging
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_constant_schedule,
    DataCollatorWithPadding,
)
from accelerate import Accelerator
import argparse
from datasets import load_dataset
from utils.utils import seed_everything, get_log, make_optimizer_group, EarlyStopping
import evaluate

# CNE
# original labels -> change
# {0:'entailment', 1:'neutral', 2:'contradiction'}
LABEL_SWITCH = {0: 2, 1: 1, 2: 1}


def get_args():
    # parser
    parser = argparse.ArgumentParser()
    # input data
    group = parser.add_argument_group(title="input data")
    group.add_argument("--data_name", type=str, help="data_name", required=True)

    # logging 관련
    group = parser.add_argument_group(title="logs")
    group.add_argument("--logging_term", type=int, default=100)

    # output
    group = parser.add_argument_group(title="output")
    group.add_argument("--output_dir", type=str, required=True)

    # preprocess
    group = parser.add_argument_group(title="preprocess")
    group.add_argument("--num_proc", type=int, default=1)

    # 학습 관련
    group = parser.add_argument_group(title="train arguments")
    group.add_argument("--epochs", type=int, default=1)
    group.add_argument("--eval_epoch", type=int, default=1, help="term of evaluation")
    group.add_argument("--batch_size", default=8, type=int)
    group.add_argument("--eval_batch_size", default=8, type=int)
    group.add_argument("--lr", type=float, default=2e-5)
    group.add_argument("--gradient_clipping", type=float, default=1.0)
    group.add_argument("--decay", type=float, default=0.1)
    group.add_argument("--accumulation_steps", type=int, default=1)

    # PTM model
    group = parser.add_argument_group(title="model")
    group.add_argument("--plm_path", type=str)
    group.add_argument("--use_flash_attention_2", action="store_true")
    group.add_argument("--num_labels", type=int, default=3)

    # model input
    group = parser.add_argument_group(title="model_input")
    group.add_argument("--max_length", type=int)
    
    # early stop 관련
    group = parser.add_argument_group(title="early_stop")
    group.add_argument("--early_stop", action="store_true")
    group.add_argument("--early_stop_metric", type=str, default="loss")
    group.add_argument("--early_stop_metric_is_max_better", action="store_true")
    group.add_argument("--patience", type=int, default=3)
    group.add_argument("--save_model_every_epoch", action="store_true")

    args = parser.parse_args()
    return args


def get_tokenizer_and_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path, trust_remote_code=True)
    # for gpt 4 model
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"  # for classification
    
    if args.use_flash_attention_2:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.plm_path,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            num_labels=args.num_labels,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.plm_path, trust_remote_code=True, num_labels=args.num_labels
        )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def preprocess_function(examples, tokenizer):
    new_examples = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for premise, hypothesis, label in zip(
        examples["premise"], examples["hypothesis"], examples["label"]
    ):
        tokenized = \
        tokenizer("Premise: " + premise + "\n\nHypothesis: " + hypothesis)
        new_examples["input_ids"].append(tokenized["input_ids"])
        new_examples["attention_mask"].append(tokenized["attention_mask"])
        _label = LABEL_SWITCH[label]
        new_examples["labels"].append(_label)
    return new_examples


def get_dataset(args, tokenizer):
    dataset = load_dataset(args.data_name)
    column_names = dataset["train"].column_names
    num_proc = args.num_proc  # int(os.cpu_count()*0.95)
    with accelerator.main_process_first():
        dataset = dataset.map(
            lambda example: preprocess_function(example, tokenizer),
            batched=True,
            num_proc=num_proc,
            remove_columns=column_names,
        )
    return dataset


def load_dataloaders(args, dataset, tokenizer):
    # LOAD DATASETS
    collate_fn = DataCollatorWithPadding(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )
    train_dataset = dataset["train"]
    val_dataset = dataset["validation_mismatched"]
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=args.eval_batch_size, collate_fn=collate_fn
    )
    return train_dataset, train_dataloader, val_dataset, val_dataloader


# evaluation
def evaluation(args, accelerator, model, tokenizer, eval_dataloader):
    accuracy = evaluate.load("accuracy")
    total_loss = 0.0
    model.eval()
    predicts = []
    actuals = []
    losses = []
    with torch.no_grad():
        for data in tqdm(
            eval_dataloader, desc="evaluate", \
            disable=not accelerator.is_main_process
        ):
            data = data.to(accelerator.device)
            output = model.forward(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                labels=data["labels"],
            )
            loss = output.loss
            total_loss += loss
            predicts.extend(output.logits.argmax(dim=-1))
            actuals.extend(data.labels.cpu())
        total_loss = total_loss / len(eval_dataloader)
        if accelerator.distributed_type != "NO":
            losses = accelerator.gather(total_loss)
            predicts = accelerator.gather(
                torch.tensor(predicts, device=accelerator.device)
            )
            actuals = accelerator.gather(
                torch.tensor(actuals, device=accelerator.device)
            )

        else:
            losses = torch.tensor([total_loss])
            predicts = torch.tensor(predicts, device=accelerator.device)
            actuals = torch.tensor(actuals, device=accelerator.device)
    _loss = losses.sum().item() / len(losses)
    _acc = accuracy.compute(
        predictions=predicts.cpu().tolist(), references=actuals.cpu().tolist()
    )
    score = dict(loss=_loss, acc=_acc)
    return score


def train():
    if accelerator.is_main_process:
        early_stop = EarlyStopping(
            args.patience,
            args.output_dir,
            max=args.early_stop_metric_is_max_better,
            min_difference=1e-5,
            model_save_dict=False,
        )
    flag_tensor = torch.zeros(1).cuda()
    ###########################################################################
    # train
    ###########################################################################
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(1, args.epochs + 1):
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0.0
        step = 0
        iter_bar = tqdm(
            train_dataloader, desc="step", \
            disable=not accelerator.is_main_process
        )
        for data in iter_bar:
            step += 1
            data = {i: j.to(accelerator.device) for i, j in data.items()}
            out = model.forward(
                input_ids=data["input_ids"],
                attention_mask=data["attention_mask"],
                labels=data["labels"],
            )
            loss = out.loss
            loss = loss / args.accumulation_steps
            accelerator.backward(loss)
            if step % args.accumulation_steps == 0 or (
                len(train_dataloader) <= args.accumulation_steps
                and (step) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clipping
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            epoch_loss += loss.mean().item() * args.accumulation_steps
            iter_bar.set_postfix(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "epoch_loss": f"{epoch_loss/step:.5f}",
                }
            )
            if args.logging_term is not None:
                if global_step % args.logging_term == 0:
                    if accelerator.is_main_process:
                        logger1.info(iter_bar)
                        logger2.info(iter_bar)

        #epoch 당 기록.
        if accelerator.is_main_process:
            logger1.info(iter_bar)
            logger2.info(iter_bar)
        #######################################################################
        #evaluation
        #######################################################################
        if args.eval_epoch != 0 and epoch % args.eval_epoch == 0:
            scores = evaluation(args, accelerator, model, \
                                tokenizer, val_dataloader)
            torch.cuda.empty_cache()

            if accelerator.is_main_process:
                logger1.info(f"Val ---- epoch : {epoch} ----- scores:{scores}")
                logger2.info(f"Val ---- epoch : {epoch} ----- scores:{scores}")
                upwrapped_model = accelerator.unwrap_model(model)
                if args.save_model_every_epoch:
                    save_path = os.path.join(args.output_dir,\
                                             "model_%d" % epoch)
                    os.makedirs(save_path, exist_ok=True)
                    upwrapped_model.save_pretrained(
                        save_path,
                        save_function=accelerator.save,
                        state_dict=accelerator.get_state_dict(model),
                    )
                    tokenizer.save_pretrained(save_path)
                early_stop.check(upwrapped_model, scores[args.early_stop_metric])
                if early_stop.timetobreak:
                    flag_tensor += 1
        accelerator.wait_for_everyone()
        
        if args.early_stop:
            to_stop = sum(accelerator.gather(flag_tensor)).item()
            if to_stop >= 1:
                if accelerator.is_main_process:
                    logger1.info("early stop")
                    logger2.info("early stop")
                    save_path = os.path.join(args.output_dir, "best_model")
                    os.makedirs(save_path, exist_ok=True)
                    early_stop.best_model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logger1.info("train_end")
                    logger2.info("train end")
                break


if __name__ == "__main__":
    # prepare
    args = get_args()
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)
    logger1, logger2 = get_log(args)

    # args.local_rank = int(os.environ["LOCAL_RANK"])

    ###########################################################################
    # tokenizer, model load
    ###########################################################################
    tokenizer, model = get_tokenizer_and_model(args)
    ###########################################################################

    ###########################################################################
    # accelerator
    ###########################################################################
    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation_steps)

    model.to(accelerator.device)
    if accelerator.is_main_process:
        logger1.info(args)
        logger2.info(args)

    # save
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "args.txt"), "w") as f:
            json.dump(args.__dict__, f, indent=2)
    ###########################################################################

    ###########################################################################
    # data
    ###########################################################################
    dataset = get_dataset(args, tokenizer)
    ###########################################################################

    ###########################################################################
    # dataloaders
    ###########################################################################
    train_dataset, train_dataloader, val_dataset, val_dataloader = \
    load_dataloaders(
        args, dataset, tokenizer
    )
    ###########################################################################

    # ###########################################################################
    # # optimizer, scheduler, synchronize
    # ###########################################################################
    optimizer_grouped_parameters = make_optimizer_group(model, args.decay)
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        optimizer_grouped_parameters, lr=args.lr, weight_decay=args.decay
    )
    scheduler = get_constant_schedule(optimizer)
    # ###########################################################################

    # ###########################################################################
    # # prepare
    # ###########################################################################
    model, optimizer, train_dataloader, val_dataloader, scheduler = \
    accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, scheduler
    )
    # ###########################################################################

    # ###########################################################################
    # # train
    # ###########################################################################
    train()
    print("done")
    # ###########################################################################
