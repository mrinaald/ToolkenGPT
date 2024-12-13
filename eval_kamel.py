import json
import os
from collections import Counter


# import chain
from itertools import chain
from collections import Counter
def check_acc(outputs, targets):
    mul_ans = 0
    c = Counter()
    for gen, tar in zip(outputs, targets):
        #ans = tar["answer"][0]
        if len(tar["answer"]) > 1:
            mul_ans += 1
        objs = [ans["alternative"] + [ans["chosen"]] for ans in tar["answer"]]
        objs = list(chain.from_iterable(objs))
        rel = tar["api"]
        generation = gen["generation"]
        # print(objs)
        if rel in generation:
            c.update(["Correct-rel"])
        elif "<P" in generation:
            c.update(["False-rel"])
        elif any([obj in generation for obj in objs]):
            c.update(["Correct-obj"])
        else:
            c.update(["False-obj"])
    # print(f"Multiple answers: {mul_ans}")
    # calculate the accuracy
    acc = (c["Correct-rel"] + c["Correct-obj"]) / (c["Correct-rel"] + c["Correct-obj"] + c["False-rel"] + c["False-obj"])
    return c, acc


##################################################
### Single task training
##################################################
# output_file = "outputs/kamel_30/inference-1B-epoch-0-kamel_embedding_inference-kamel_30-bias_10.jsonl"
# target_file = "data/kamel/test_first_30.json"

# output_file = "outputs/kamel_60/inference-1B-epoch-0-kamel_embedding_inference-kamel_60-bias_10.jsonl"
# target_file = "data/kamel/test_first_60.json"

# output_file = "outputs/kamel_100/inference-1B-epoch-0-kamel_embedding_inference-kamel_100-bias_10.jsonl"
# target_file = "data/kamel/test_first_100.json"

# output_file = "outputs/kamel_234/inference-1B-epoch-0-kamel_embedding_inference-kamel_234-bias_10.jsonl"
# target_file = "data/kamel/test_first_234.json"


##################################################
### Multi-task training
##################################################
# output_file = "outputs/kamel_30/multitask-inference-1B-epoch_0-kamel_embedding_inference-kamel_30-bias_10.jsonl"
# target_file = "data/kamel/test_first_30.json"

# output_file = "outputs/kamel_60/multitask-inference-1B-epoch_0-kamel_embedding_inference-kamel_60-bias_10.jsonl"
# target_file = "data/kamel/test_first_60.json"

# output_file = "outputs/kamel_100/multitask-inference-1B-epoch_0-kamel_embedding_inference-kamel_100-bias_10.jsonl"
# target_file = "data/kamel/test_first_100.json"

output_file = "outputs/kamel_234/multitask-inference-1B-epoch_0-kamel_embedding_inference-kamel_234-bias_10.jsonl"
target_file = "data/kamel/test_first_234.json"


outputs = []
#with open("outputs/inference-13B-ood_para_kamel_embedding.pt-func_embedding-kamel-0.jsonl") as f:
#    for line in f:
#        outputs.append(json.loads(line))
with open(output_file) as f:
    for line in f:
        outputs.append(json.loads(line))
target = json.load(open(target_file))
print(check_acc(outputs, target))
