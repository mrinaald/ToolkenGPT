# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
import sys
import torch
#import fire
import time
import json
import random
#import wandb
import numpy as np
from tqdm import tqdm
from typing import Tuple
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from transformers import AutoTokenizer, AutoModelForCausalLM

from llama import ModelArgs, Transformer, Tokenizer, FunctionLM

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


# def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, func_dict: dict) -> FunctionLM:
def load(model_uri: str, local_rank: int, world_size: int, func_dict: dict, func_emb_file: str = None) -> FunctionLM:
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
    # model = AutoModelForCausalLM.from_pretrained(model_uri, device_map="cuda", load_in_8bit=True)

    funcmodel = FunctionLM(model, tokenizer, func_dict = func_dict, load_path=func_emb_file)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return funcmodel


def main(model_uri: str, input_file: str = None, lr: float = 1e-3, num_epochs: int = 20, dataset: str = "gsm8k-xl", log_prefix="", only_functoken=False, log_each=False, debug=False, func_emb_file: str = None):

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)

    # new token -> index
    func_dict_path = f"data/{dataset}/func_dict.json"

    func_dict = json.load(open(func_dict_path, "r"))

    # local_rank, world_size = setup_model_parallel()
    local_rank, world_size = 0, 1
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    if local_rank == 0:
        # wandb.init(project="funcllama", name=f"{dataset}-{world_size}-load")
        # wandb.init(project="opt", name=save_name)
        print(f"Project: funcllama | name {dataset}-{world_size}-load")

    # funcmodel = load(ckpt_dir, tokenizer_path, local_rank, world_size, func_dict=func_dict)
    funcmodel = load(model_uri, local_rank, world_size, func_dict=func_dict, func_emb_file=func_emb_file)

    if input_file.endswith(".json"):
        with open(input_file, "r") as f:
            prompts = json.load(f)

    else:
        with open(input_file, "r") as f:
            prompts = f.readlines()
        prompts = [prompt.strip().replace("\\n", "\n") for prompt in prompts if len(prompt) > 1]

    if dataset == "gsm8k-xl":
        # the last 1000 prompts are the testset
        test_len = 1000
    elif dataset == "funcqa":
        # the last 39 prompts are the testset
        test_len = 39
    elif dataset == "vh":
        test_len = 47
    elif dataset == "kamel":
        test_len = 1000

    log_index = 200
    batch_save_index = 500
    if debug:
        # prompts = prompts[:len(func_dict) * 5]
        prompts = prompts[:100]
        test_len = int(len(prompts) * 0.2)
        log_index = 20
        batch_save_index = 20

    testset = prompts[-test_len:]
    trainset = prompts[:-test_len]

    # only update tokens with gradients required
    trainable_params = [p for p in funcmodel.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    print(f"Num trainable parameter matrices/vectors: {len(trainable_params)}")
    print(f"Total trainable parameters: {sum([p.numel() for p in trainable_params])}")

    decayRate = 0.1
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    # func_dict
    func_dict = funcmodel.func_dict
    func_list = list(func_dict.keys())

    from collections import defaultdict
    for epoch in range(num_epochs):
        results = defaultdict(list)

        random.shuffle(trainset)
        for case_idx, prompt in tqdm(enumerate(trainset), total=len(trainset), desc=f"Epoch: {epoch + 1}/{num_epochs}"):
            funcmodel.train()

            optimizer.zero_grad()
            loss, result = funcmodel.get_loss([prompt], only_functoken=only_functoken)
            loss.backward()
            optimizer.step()

            for i, r in result.items():
                results[i].append(r)

            if (case_idx + 1) % log_index == 0:
                for i in range(len(func_list)+1):
                    if i != len(func_list):
                        tp, pred, true = sum([r[i] for r in results["tp"]]), sum([r[i] for r in results["pred"]]), sum([r[i] for r in results["true"]])
                    else:
                        tp, pred, true = sum([r.sum() for r in results["tp"]]), sum([r.sum() for r in results["pred"]]), sum([r.sum() for r in results["true"]])
                    # tqdm.write(f"tp: {tp}, pred: {pred}, true: {true}")

                    if local_rank == 0:
                        if i != len(func_list) and log_each:
                            tqdm.write("")
                            # wandb.log({
                            #     f"precision-{i}": tp / (pred + 1e-8),
                            #     f"recall-{i}": tp / (true + 1e-8),
                            #     f"f1-{i}": 2 * tp / (pred + true + 1e-8)
                            # })
                            log_data = {
                                f"tp-{i}": tp,
                                f"pred-{i}": pred,
                                f"true-{i}": true,
                                f"precision-{i}": tp / (pred + 1e-8),
                                f"recall-{i}": tp / (true + 1e-8),
                                f"f1-{i}": 2 * tp / (pred + true + 1e-8)
                            }
                            tqdm.write(f"{log_data}")
                            # tqdm.write(f"tp-{i}: {tp:.6f}")
                            # tqdm.write(f"pred-{i}: {pred:.6f}")
                            # tqdm.write(f"true-{i}: {true:.6f}")
                            # tqdm.write(f"precision-{i}: {tp / (pred + 1e-8):.6f}")
                            # tqdm.write(f"recall-{i}: {tp / (true + 1e-8):.6f}")
                            # tqdm.write(f"f1-{i}: {tp / (pred + true + 1e-8):.6f}")
                            tqdm.write("")
                        elif i == len(func_list):
                            tqdm.write("")
                            # wandb.log({
                            #     f"precision": tp / (pred + 1e-8),
                            #     f"recall": tp / (true + 1e-8),
                            #     f"f1": 2 * tp / (pred + true + 1e-8)
                            # })
                            log_data = {
                                f"tp": tp,
                                f"pred": pred,
                                f"true": true,
                                f"precision": tp / (pred + 1e-8),
                                f"recall": tp / (true + 1e-8),
                                f"f1": 2 * tp / (pred + true + 1e-8)
                            }
                            tqdm.write(f"{log_data}")
                            # tqdm.write(f"tp: {tp:.6f}")
                            # tqdm.write(f"pred: {pred:.6f}")
                            # tqdm.write(f"true: {true:.6f}")
                            # tqdm.write(f"precision: {tp / (pred + 1e-8):.6f}")
                            # tqdm.write(f"recall: {tp / (true + 1e-8):.6f}")
                            # tqdm.write(f"f1: {tp / (pred + true + 1e-8):.6f}")
                            tqdm.write("")
                        # save the parameters of func_embed
                        # torch.save(funcmodel.func_embed.state_dict(), save_file)
                results = defaultdict(list)

            if local_rank == 0:
                # wandb.log({"loss": loss.item()})
                tqdm.write(f"loss: {loss.item()}")

            if (case_idx + 1) % batch_save_index == 0:
                save_dir = f"checkpoints/{dataset}{log_prefix}/"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(funcmodel.func_embed.state_dict(), f"{save_dir}/epoch_{epoch}_{case_idx + 1}.pth")

        # test on validation set
        results = defaultdict(list)
        for case_idx, prompt in tqdm(enumerate(testset), total=len(testset), desc="Evaluating"):
            funcmodel.eval()
            with torch.no_grad():
                loss, result = funcmodel.get_loss([prompt])

            for i, r in result.items():
                results[i].append(r)

        for i in range(len(func_list) + 1):
            if i != len(func_list):
                tp, pred, true = sum([r[i] for r in results["tp"]]), sum([r[i] for r in results["pred"]]), sum([r[i] for r in results["true"]])
            else:
                # 4 is for all functions
                tp, pred, true = sum([r.sum() for r in results["tp"]]), sum([r.sum() for r in results["pred"]]), sum([r.sum() for r in results["true"]])
            # print(f"tp: {tp}, pred: {pred}, true: {true}")

            if local_rank == 0:
                if i != len(func_list) and log_each:
                    print()
                    # wandb.log({
                    #     f"test-precision-{i}": tp / (pred + 1e-8),
                    #     f"test-recall-{i}": tp / (true + 1e-8),
                    #     f"test-f1-{i}": 2 * tp / (pred + true + 1e-8)
                    # })
                    print({
                        f"test-tp-{i}": tp,
                        f"test-pred-{i}": pred,
                        f"test-true-{i}": true,
                        f"test-precision-{i}": tp / (pred + 1e-8),
                        f"test-recall-{i}": tp / (true + 1e-8),
                        f"test-f1-{i}": 2 * tp / (pred + true + 1e-8)
                    })
                    # print(f"test-tp-{i}: {tp:.6f}")
                    # print(f"test-pred-{i}: {pred:.6f}")
                    # print(f"test-true-{i}: {true:.6f}")
                    # print(f"test-precision-{i}: {tp / (pred + 1e-8):.6f}")
                    # print(f"test-recall-{i}: {tp / (true + 1e-8):.6f}")
                    # print(f"test-f1-{i}: {tp / (pred + true + 1e-8):.6f}")
                    print()
                elif i == len(func_list):
                    print()
                    # wandb.log({
                    #     f"test-precision": tp / (pred + 1e-8),
                    #     f"test-recall": tp / (true + 1e-8),
                    #     f"test-f1": 2 * tp / (pred + true + 1e-8)
                    # })
                    print({
                        f"test-tp": tp,
                        f"test-pred": pred,
                        f"test-true": true,
                        f"test-precision": tp / (pred + 1e-8),
                        f"test-recall": tp / (true + 1e-8),
                        f"test-f1": 2 * tp / (pred + true + 1e-8)
                    })
                    # print(f"test-tp: {tp:.6f}")
                    # print(f"test-pred: {pred:.6f}")
                    # print(f"test-true: {true:.6f}")
                    # print(f"test-precision: {tp / (pred + 1e-8):.6f}")
                    # print(f"test-recall: {tp / (true + 1e-8):.6f}")
                    # print(f"test-f1: {tp / (pred + true + 1e-8):.6f}")
                    print()
        # lr_scheduler.step()

        # save the parameters of func_embed every epoch
        save_dir = f"checkpoints/{dataset}{log_prefix}/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(funcmodel.func_embed.state_dict(), f"{save_dir}/epoch_{epoch}.pth")
        results = defaultdict(list)

if __name__ == "__main__":
    # fire.Fire(main)
    MODEL = "meta-llama/Llama-3.2-1B"

    # dataset: str = "gsm8k-xl"
    # input_file: str = "data/gsm8k-xl/Llama-3.2-1B_gsm8k_train.json"
    # lr: float = 1e-4
    # num_epochs: int = 2
    # only_functoken=False
    # func_emb_file = "checkpoints/gsm8k-xl_20241122_1510_lr1e-4_e5/epoch-0_iter-5000.pth"
    # log_prefix="-20241211_1810"
    # # log_each=False

    # dataset: str = "funcqa"
    # input_file: str = "data/funcqa/Llama-3.2-1B-funcqa_train.json"
    # lr: float = 1e-3
    # num_epochs: int = 10
    # only_functoken=False
    # func_emb_file = None
    # # log_prefix=""
    # # log_each=False

    # dataset: str = "kamel"
    # input_file: str = "data/kamel/Llama-3.2-1B-kamel_id_train.json"       # supervised data
    # input_file: str = "data/kamel/Llama-3.2-1B-kamel_train_clean.json"    # synthetic data
    # lr: float = 3e-4        # 1e-4 worked for non-quantized version
    # num_epochs: int = 3
    # only_functoken=False
    # func_emb_file = None
    # log_prefix=""
    # log_each=False

    dataset: str = "vh"
    input_file = "data/vh/Llama-3.2-1B_vh_legal_train_v4_embedding.json"
    lr = 1e-3
    num_epochs = 5
    only_functoken = True
    func_emb_file = None
    log_prefix = "-debug"

    print(f"dataset: [{dataset}]")
    print(f"input_file: [{input_file}]")
    print(f"lr: [{lr}]")
    print(f"num_epochs: [{num_epochs}]")
    print(f"only_functoken: [{only_functoken}]")

    main(
        model_uri=MODEL,
        dataset=dataset,
        input_file=input_file,
        lr=lr,
        num_epochs=num_epochs,
        only_functoken=only_functoken,
        log_each=False,
        log_prefix=log_prefix,
        func_emb_file=func_emb_file,
        debug=True,
    )
