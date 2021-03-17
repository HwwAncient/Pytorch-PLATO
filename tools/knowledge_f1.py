"""
Calculate Knowledge f1.
"""

import sys
import json

import numpy as np

eval_file = sys.argv[1]
test_file = sys.argv[2]

cnt = 0
res = 0.0
r = 0.0
p = 0.0
stopwords = set()
with open("./tools/stopwords.txt") as f:
    for line in f:
        word = line.strip()
        stopwords.add(word)

with open(eval_file) as f:
    for result, line in zip(json.load(f), open(test_file)):
        cnt += 1
        if "scores" in result:
            pred = result["preds"][np.argmax(result["scores"])]
        else:
            pred = result["preds"][0]
        knowledges, _, reply = line.strip().split('\t')

        words = set()
        for sent in knowledges.split(" __eou__ "):
            for word in sent.split():
                words.add(word)
        words = words - stopwords
        k_len = len(words)

        pred1 = set(pred.split())
        pred1 = pred1 - stopwords
        pred_len = len(pred1)
        overlap = len(words & pred1)

        if overlap == 0:
            continue

        recall = float(overlap) / k_len
        r += recall
        precison = float(overlap) / pred_len
        p += precison
        res += 2*recall*precison/(recall+precison)
print(f"Recall:{r/cnt}")
print(f"Precison:{p/cnt}")
print(f"F1:{res/cnt}")
print("Recall/Precision/F1:{:0,.4f}/{:0,.4f}/{:0,.4f}".format(r/cnt, p/cnt, res/cnt))

