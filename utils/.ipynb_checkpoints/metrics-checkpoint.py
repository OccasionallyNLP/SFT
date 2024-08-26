# -*- coding: utf-8 -*-
# metrics
import numpy as np
from collections import Counter
import string
import re
import argparse
import json
import sys
import os
from typing import List
from rouge import Rouge
import numpy as np

# aggregation
def get_scores(predictions:List[str], actuals:List[List[str]]):
    scores = {}
    total_f1 = []
    total_em = []
    for predict, actual in zip(predictions, actuals):
        f1 = metric_max_over_ground_truths(f1_score, predict, actual)
        em = metric_max_over_ground_truths(exact_match_score, predict, actual)
        total_f1.append(f1)
        total_em.append(em)
    scores['f1']=sum(total_f1)/len(total_f1)
    scores['em']=sum(total_em)/len(total_em)
    return scores

# get max
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

# answer nomalization
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# em
def exact_match_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    answer = normalize_answer(ground_truth)
    if not answer:
        print('some error in here.')
        return False
    return prediction == answer

# token unigram f1 score
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def repetition_4(prediction):
    prediction_tokens = normalize_answer(prediction).split()
    
    
def rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def bleu_score(prediction, ground_truth):
    hypothesis = prediction.split()
    reference = ground_truth.split()
    cc = SmoothingFunction()
    score1 = sentence_bleu([reference],hypothesis,weights=(1,0,0,0))#, smoothing_function = cc.method4)
    score4 = sentence_bleu([reference],hypothesis,weights=(0,0,0,1), smoothing_function = cc.method4)
    return score1, score4
