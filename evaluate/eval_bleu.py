#coding:utf-8
import json
import sys
import evaluate

tgt_key = "sentence"
# tgt_key = "tgt"
with open(sys.argv[1], 'r', encoding='utf-8') as fin:
    gold = [json.loads(itm.strip()) for itm in fin.readlines()]
    gold_label = [[itm[tgt_key]] for itm in gold]
    
with open(sys.argv[2], 'r', encoding='utf-8') as fin:
    pred_data = [itm.strip() for itm in fin.readlines()]
    
# metric = evaluate.load("sacrebleu")
metric = evaluate.load(path="./bleu.py")
result = metric.compute(predictions=pred_data, references=gold_label, lowercase=True)
print(result)