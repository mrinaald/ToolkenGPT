import json
import re
from funchub.math import *

def parse_answer(answer, pattern:str="####"):
    if pattern=="####":
        answer = answer.split("####")[-1]
        answer = answer.strip().strip("\n").strip('\\n')
        # 32,333 -> 32333
        answer = answer.replace(",", "")

        # get the last number
        try:
            answer = re.findall(r"[-+]?\d*\.\d+|\d+", answer)[-1]
        except:
            answer = 0
    elif pattern=="answer is":
        answer = answer.split("answer is")[-1]
        answer = answer.strip().strip("\n").strip('\\n')

        # 32,333 -> 32333
        answer = answer.replace(",", "")

        # get the last number
        try:
            answer = re.findall(r"[-+]?\d*\.\d+|\d+", answer)[-1]
        except:
            answer = 0

    return answer

def accuracy(pred, true, type = "exact"):
    if len(pred) < len(true):
        true = true[:len(pred)]

    correct = 0
    for p, t in zip(pred, true):
        try:
            if type == "exact":
                if float(p) == float(t):
                    correct += 1
            elif type == "round":
                if round(float(p), 2) == custom_round(float(t), 2):
                    correct += 1
            elif type == "approx":
                # 1% error tolerance, e.g. 1000 -> 990 ~ 1010
                if abs(float(p) - float(t)) <= abs(float(t)) * 0.001:
                    correct += 1
        except ValueError:
            pass

    return correct / len(pred)



target_path = "data/gsm8k-xl/test.json"
eval_path = "outputs/gsm8k-xl/inference-1B-epoch-0_iter-5000-func_embedding-gsm8k-xl-bias_3.0.jsonl"
# target_path = "data/funcqa/funcqa_oh.json"
# eval_path = "outputs/funcqa_oh/inference-1B-epoch_1-func_embedding-funcqa_oh-bias_2.7.jsonl"
# target_path = "data/funcqa/funcqa_mh.json"
# eval_path = "outputs/funcqa_mh/inference-7B-None-baseline-funcqa_mh-0.jsonl"


if "gsm8k" in target_path:
    with open(target_path, "r") as f:
        data = [json.loads(line) for line in f.readlines()]
else:
    with open(target_path, "r") as f:
        data = json.load(f)

if "gsm8k" in target_path:
    answer = [d["enhanced_result"] for d in data]
else:
    answer = [d["answer"] for d in data]

with open(eval_path, "r") as f:
    data = [json.loads(line) for line in f]

pred = [parse_answer(d["generation"], pattern="####") for d in data]

print(pred)
print(answer[:len(pred)])

print("Accuracy: ", accuracy(pred, answer[:len(pred)], type="approx"))
