# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
# import fire
import time
import json
import re
import random
import numpy as np
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm
from llama import ModelArgs, Transformer, Tokenizer, FunctionLM
from inference_modes import func_embedding_inference, kamel_embedding_inference, vh_embedding_inference
from funchub.math import *

from transformers import AutoTokenizer, AutoModelForCausalLM

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


# def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, func_load_path: str, func_dict: dict) -> FunctionLM:
def load(model_uri: str, local_rank: int, world_size: int, func_dict: dict, func_load_path: str) -> FunctionLM:
    start_time = time.time()
    # checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    # assert (
    #     world_size == len(checkpoints)
    # ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    # ckpt_path = checkpoints[local_rank]
    # print("Loading")
    # checkpoint = torch.load(ckpt_path, map_location="cpu")

    # with open(Path(ckpt_dir) / "params.json", "r") as f:
    #     params = json.loads(f.read())

    # model_args: ModelArgs = ModelArgs(max_seq_len=2048, max_batch_size=1, **params)
    # tokenizer = Tokenizer(model_path=tokenizer_path)
    # model_args.vocab_size = tokenizer.n_words
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
    # model = Transformer(model_args).cuda().half()
    # torch.set_default_tensor_type(torch.FloatTensor)
    # model.load_state_dict(checkpoint, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(model_uri)
    model = AutoModelForCausalLM.from_pretrained(model_uri)
    ## pip install bitsandbytes accelerate
    # model = AutoModelForCausalLM.from_pretrained(model_uri, device_map="cuda", load_in_8bit=True)
    # model = AutoModelForCausalLM.from_pretrained(model_uri, device_map="cuda", load_in_4bit=True)

    funcmodel = FunctionLM(model, tokenizer, func_dict = func_dict, load_path=func_load_path)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return funcmodel


# def main(ckpt_dir: str, tokenizer_path: str, temperature: float = 0, top_p: float = 0.95, mode: str = "baseline", dataset = "original", return_top: int = 5, logits_bias: float = 0, func_load_path: str = "None", st_idx=0, ed_idx=10000, suffix=""):
def main(model_uri: str, temperature: float = 0, top_p: float = 0.95, mode: str = "baseline", dataset = "original", return_top: int = 5, logits_bias: float = 0, func_load_path: str = "None", st_idx=0, ed_idx=10000, suffix=""):
    # set random seed
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)
    # size = ckpt_dir.split("/")[-1]
    # local_rank, world_size = setup_model_parallel()
    # if local_rank > 0:
    #     sys.stdout = open(os.devnull, 'w')
    size = model_uri.split("-")[-1]
    local_rank, world_size = 0, 0

    templates = {}
    if dataset == "gsm8k-xl":
        for name in os.listdir("data/gsm8k-xl/template"):
            with open(f"data/gsm8k-xl/template/{name}") as f:
                if "__" in name:
                    # "<" and ">" in filenames got converted to "_" in Windows
                    templates["<" + name.split("_")[2] + ">"] = f.read()
                else:
                    templates[name.split("_")[-1].replace(".txt", "")] = f.read()
        with open(f"data/gsm8k-xl/test.json") as f:
            data = [json.loads(line) for line in f.readlines()]
        raw_test_cases = [i["question"] for i in data]
        enhanced_v = [i["enhanced_v"] for i in data]
        test_cases = []
        for v, q in zip(enhanced_v, raw_test_cases):
            for i in range(len(v)):
                q = q.replace(f"{{v_{i+1}}}", str(v[i]))
            test_cases.append(q)

        max_gen_len = 512
        func_dict = json.load(open("data/gsm8k-xl/func_dict.json"))

    elif dataset == "funcqa_mh":
        for name in os.listdir("data/funcqa/template_mh"):
            with open(f"data/funcqa/template_mh/{name}") as f:
                if "__" in name:
                    # "<" and ">" in filenames got converted to "_" in Windows
                    templates["<" + name.split("_")[2] + ">"] = f.read()
                else:
                    templates[name.split("_")[-1].replace(".txt", "")] = f.read()
        with open("data/funcqa/funcqa_mh.json") as f:
            data = json.load(f)
        test_cases = [i["question"] for i in data]
        max_gen_len = 512
        func_dict = json.load(open("data/funcqa/func_dict.json"))

    elif dataset == "funcqa_oh":
        for name in os.listdir("data/funcqa/template_oh"):
            with open(f"data/funcqa/template_oh/{name}") as f:
                if "__" in name:
                    # "<" and ">" in filenames got converted to "_" in Windows
                    templates["<" + name.split("_")[2] + ">"] = f.read()
                else:
                    templates[name.split("_")[-1].replace(".txt", "")] = f.read()
        with open("data/funcqa/funcqa_oh.json") as f:
            data = json.load(f)
        max_gen_len = 512
        func_dict = json.load(open("data/funcqa/func_dict.json"))
        test_cases = [i["question"] for i in data]

    elif dataset == "vh":
        from vh_eval import get_desc
        assert mode in ["vh_embedding_inference", "baseline"]
        with open("data/vh/legal_test_v2.json") as f:
            file_list = json.load(f)
        with open("data/vh/func_dict.json") as f:
            func_dict = json.load(f)

        if mode == "vh_embedding_inference":
            test_cases = []
            with open("data/vh/template/vh_special_v4.txt") as f:
                template = f.read()
            existing_obj_list = []

            for fun in func_dict:
                if fun.startswith("<"):
                    existing_obj_list.append(fun[1:-1])

            for script_file, state_file in file_list:
                with open(script_file) as f:
                    script = f.read()
                    title = script.split("\n")[0]
                    goal = script.split("\n")[1]
                    desc = get_desc(graph_file_name=state_file, script_file_name=script_file, obj_list=existing_obj_list)
                    obj_list = re.search(r"The objects I can manipulate are (.*?)\.", desc).group(1)
                    obj_list = eval(obj_list)
                    obj_list = [f"<{o}>" for o in obj_list]
                    discard_list = [o for o in func_dict if o not in obj_list and o.startswith("<")]
                    test_cases.append((template.replace("[QUESTION]", desc), discard_list))

            print(test_cases[0][0]+"[START]")
            print(test_cases[0][1])

        max_gen_len = 96
        max_func_call = 32


    elif dataset.startswith("kamel"):
        n_first = int(dataset.split("_")[-1])
        for name in os.listdir("data/kamel/template"):
            with open(f"data/kamel/template/{name}") as f:
                templates[name.split("_")[-1].replace(".txt", "")] = f.read()

        with open(f"data/kamel/test_first_{n_first}.json") as f:
            data = json.load(f)
            test_cases = [i["question"] for i in data]
        # func_dict = {f"<{r}>": ind for ind, r in enumerate(func_dict)}
        func_dict = json.load(open("data/kamel/func_dict.json"))
        func_dict = {f"<{k}>": v for k, v in func_dict.items()}
        func_dict = {k: v for k, v in func_dict.items() if v < n_first}
        print(len(func_dict))
        max_gen_len = 30
        max_func_call = 1


    # funcmodel = load(ckpt_dir, tokenizer_path, local_rank, world_size, func_load_path=func_load_path, func_dict=func_dict)
    funcmodel = load(model_uri, local_rank, world_size, func_load_path=func_load_path, func_dict=func_dict)
    funcmodel.set_bias(logits_bias)
    funcmodel.eval()

    # test_cases = test_cases[:10]

    print(f"Mode: {mode}")
    print(f"Dataset: {dataset}")
    print(f"Logits Bias: {logits_bias}")

    if local_rank == 0:
        output_dir = f"outputs/{dataset}"
        os.makedirs(output_dir, exist_ok=True)

        try:
            func_model_name = func_load_path.split('/')[-1].split('.')[0]
        except:
            func_model_name = func_load_path

        output_file = f"{output_dir}/inference-{size}-{func_model_name}-{mode}-{dataset}-bias_{logits_bias}{suffix}.jsonl"
        if os.path.exists(output_file):
            f = open(output_file, "w")
            f.close()

    for case_idx, question in tqdm(enumerate(test_cases), total=len(test_cases)):
        if case_idx < st_idx:
            continue
        if case_idx >= ed_idx:
            break
        if mode == "func_embedding":
            log = func_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top)
        elif mode == "vh_embedding_inference":
            log = vh_embedding_inference(case_idx, question, funcmodel, temperature, top_p, max_func_call)
        elif mode == "kamel_embedding_inference":
            log = kamel_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, max_func_call)



        if local_rank == 0:
            with open(output_file, "a") as f:
                f.write(json.dumps(log) + "\n")


if __name__ == "__main__":
    # fire.Fire(main)

    MODEL = "meta-llama/Llama-3.2-1B"

    """
    inference_llama.py
    --ckpt_dir $LLAMA_CKPTS/30B
    --tokenizer_path $LLAMA_CKPTS/tokenizer.model
    --mode func_embedding
    --dataset gsm8k-xl
    --func_load_path checkpoints/gsm8k-xl/epoch_3.pth
    --logits_bias 3.0
    """

    """
    inference_llama.py
    --ckpt_dir $LLAMA_CKPTS/30B
    --tokenizer_path $LLAMA_CKPTS/tokenizer.model
    --mode func_embedding
    --dataset funcqa_oh
    --func_load_path checkpoints/funcqa/epoch_7.pth
    --logits_bias 2.7
    """

    """
    inference_llama.py
    --ckpt_dir $LLAMA_CKPTS/13B
    --tokenizer_path $LLAMA_CKPTS/tokenizer.model
    --mode kamel_embedding_inference
    --dataset kamel_30
    --func_load_path checkpoints/kamel/epoch_4.pth
    --logits_bias 10
    """

    # mode = "func_embedding"
    # dataset = "gsm8k-xl"
    # func_load_path = "checkpoints/gsm8k-xl_20241122_1510_lr1e-4_e5/epoch-0_iter-5000.pth"
    # logits_bias = 3.0


    # mode = "func_embedding"
    # dataset = "funcqa_oh"
    # func_load_path = "checkpoints/funcqa-siddhant/epoch_1.pth"
    # logits_bias = 2.7

    mode = "kamel_embedding_inference"
    dataset = "kamel_234"
    func_load_path = "checkpoints/kamel_20241122_1225_lr1e-4_e2/epoch-0.pth"
    logits_bias = 10

    main(
        model_uri = MODEL,
        mode = mode,
        dataset = dataset,
        func_load_path = func_load_path,
        logits_bias = logits_bias,
    )
