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
from tqdm import tqdm, trange
from typing import List, Tuple
from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from transformers import AutoTokenizer, AutoModelForCausalLM

from llama import ModelArgs, Transformer, Tokenizer, MultiTaskFunctionLM

def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


# def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int, func_dict: dict) -> FunctionLM:
def load(model_uri: str, local_rank: int, world_size: int, func_dicts: List[dict], func_emb_file: str = None) -> MultiTaskFunctionLM:
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

    funcmodel = MultiTaskFunctionLM(model, tokenizer, func_dicts = func_dicts, load_path=func_emb_file)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return funcmodel


def main(model_uri: str, datasets: List[str], input_files: List[str] = None, lr: float = 1e-3, num_epochs: int = 20, log_suffix="", only_functoken=False, log_each=False, debug=False, func_emb_file: str = None, log_index = 200, batch_save_index = 500):

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(1)
    np.random.seed(1)

    func_dicts = []
    prompts = []
    test_lengths = []
    for dataset, input_file in zip(datasets, input_files):
        # new token -> index
        func_dict_path = f"data/{dataset}/func_dict.json"

        func_dict = json.load(open(func_dict_path, "r"))
        func_dicts.append(func_dict)

        # local_rank, world_size = setup_model_parallel()
        local_rank, world_size = 0, 1
        if local_rank > 0:
            sys.stdout = open(os.devnull, 'w')

        if local_rank == 0:
            # wandb.init(project="funcllama", name=f"{dataset}-{world_size}-load")
            # wandb.init(project="opt", name=save_name)
            print(f"Project: funcllama | name {dataset}-{world_size}-load")

        if input_file.endswith(".json"):
            with open(input_file, "r") as f:
                data_prompts = json.load(f)

        else:
            with open(input_file, "r") as f:
                data_prompts = f.readlines()
            data_prompts = [prompt.strip().replace("\\n", "\n") for prompt in data_prompts if len(prompt) > 1]

        if dataset == "gsm8k-xl":
            # the last 1000 prompts are the testset
            test_length = 1000
        elif dataset == "funcqa":
            # the last 39 prompts are the testset
            test_length = 39
        elif dataset == "vh":
            test_length = 47
        elif dataset == "kamel":
            test_length = 1000
        else:
            raise NotImplementedError(f"Unknown dataset: [{dataset}]")

        prompts.append(data_prompts)
        test_lengths.append(test_length)

    log_index = 200
    batch_save_index = 500
    if debug:
        log_index = 20
        batch_save_index = 20

        # prompts = prompts[:len(func_dict) * 5]
        for t in range(len(prompts)):
            prompts[t] = prompts[t][:100]
            test_lengths[t] = int(len(prompts[t]) * 0.2)

    trainsets = []
    testsets = []
    for t in range(len(prompts)):
        trainset = prompts[t][:-test_lengths[t]]
        testset = prompts[t][-test_lengths[t]:]
        trainsets.append(trainset)
        testsets.append(testset)

    assert len(trainsets) == len(datasets)
    assert len(trainsets) == len(func_dicts)
    assert len(trainsets) == len(testsets)

    # funcmodel = load(ckpt_dir, tokenizer_path, local_rank, world_size, func_dict=func_dict)
    funcmodel = load(model_uri, local_rank, world_size, func_dicts=func_dicts, func_emb_file=func_emb_file)

    # only update tokens with gradients required
    trainable_params = [p for p in funcmodel.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    print(f"Num trainable parameter matrices/vectors: {len(trainable_params)}")
    print(f"Total trainable parameters: {sum([p.numel() for p in trainable_params])}")

    # decayRate = 0.1
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    # # func_dict
    # func_dict = funcmodel.func_dict
    # func_list = list(func_dict.keys())
    # print([len(t) for t in trainsets])      # [5054, 19000]

    from collections import defaultdict
    for epoch in range(num_epochs):
        # results = defaultdict(list)
        results = [defaultdict(list) for _ in range(len(datasets))]

        # random.shuffle(trainset)
        for t in range(len(datasets)):
            random.shuffle(trainsets[t])

        min_dataset_size = min([len(trainset) for trainset in trainsets])
        batch_sizes = [int(len(trainset) / min_dataset_size) for trainset in trainsets]     # batch size == 1 for some dataset with min samples
        dataset_indices = [0 for _ in range(len(datasets))]

        for batch_idx in trange(min_dataset_size, desc=f"Epoch: {epoch + 1}/{num_epochs}"):
            funcmodel.train()
            optimizer.zero_grad()

            train_prompts = []
            for t in range(len(datasets)):
                trainset = [trainsets[t][j] for j in range(dataset_indices[t], dataset_indices[t] + batch_sizes[t])]
                dataset_indices[t] += batch_sizes[t]
                train_prompts.append(trainset)

            batch_loss, batch_results = funcmodel.get_loss(train_prompts, only_functoken=only_functoken)
            loss = sum(batch_loss)
            loss.backward()
            optimizer.step()

            assert len(results) == len(batch_results)
            for t in range(len(results)):
                for k, v in batch_results[t].items():
                    results[t][k].append(v)

            if (batch_idx + 1) % log_index == 0:
                for t in range(len(datasets)):
                    dataset_func_list = sorted(list(func_dicts[t].keys()))
                    dataset_results = results[t]

                    for i in range(len(dataset_func_list)+1):
                        if i != len(dataset_func_list):
                            tp, pred, true = sum([r[i] for r in dataset_results["tp"]]), sum([r[i] for r in dataset_results["pred"]]), sum([r[i] for r in dataset_results["true"]])
                        else:
                            tp, pred, true = sum([r.sum() for r in dataset_results["tp"]]), sum([r.sum() for r in dataset_results["pred"]]), sum([r.sum() for r in dataset_results["true"]])

                        if local_rank == 0:
                            if i != len(dataset_func_list) and log_each:
                                tqdm.write("")
                                log_data = {
                                    f"data-{datasets[t]}_tp-{dataset_func_list[i]}": tp,
                                    f"data-{datasets[t]}_pred-{dataset_func_list[i]}": pred,
                                    f"data-{datasets[t]}_true-{dataset_func_list[i]}": true,
                                    f"data-{datasets[t]}_precision-{dataset_func_list[i]}": tp / (pred + 1e-8),
                                    f"data-{datasets[t]}_recall-{dataset_func_list[i]}": tp / (true + 1e-8),
                                    f"data-{datasets[t]}_f1-{dataset_func_list[i]}": 2 * tp / (pred + true + 1e-8)
                                }
                                tqdm.write(f"{log_data}")
                                tqdm.write("")

                            elif i == len(dataset_func_list):
                                tqdm.write("")
                                log_data = {
                                    f"data-{datasets[t]}_tp": tp,
                                    f"data-{datasets[t]}_pred": pred,
                                    f"data-{datasets[t]}_true": true,
                                    f"data-{datasets[t]}_precision": tp / (pred + 1e-8),
                                    f"data-{datasets[t]}_recall": tp / (true + 1e-8),
                                    f"data-{datasets[t]}_f1": 2 * tp / (pred + true + 1e-8)
                                }
                                tqdm.write(f"{log_data}")
                                tqdm.write("")
                    results[t] = defaultdict(list)


            if local_rank == 0:
                tqdm.write(f"loss: {[loss.item() for loss in batch_loss]}")

            if (batch_idx + 1) % batch_save_index == 0:
                save_dir = "-".join([d.replace("-", "_") for d in datasets])
                save_dir = f"checkpoints/{save_dir}{log_suffix}/"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(funcmodel.func_embed.state_dict(), f"{save_dir}/epoch_{epoch}_{batch_idx + 1}.pth")

        # test on validation set
        funcmodel.eval()
        results = [defaultdict(list) for _ in range(len(datasets))]
        for t in range(len(datasets)):
            for batch_idx, prompt in tqdm(enumerate(testsets[t]), total=len(testsets[t]), desc=f"Evaluating [{datasets[t]}]"):
                with torch.no_grad():
                    batch_loss, batch_results = funcmodel.get_loss_for_type(prompt, toolken_type=t)

                for k, v in batch_results.items():
                    results[t][k].append(v)

            dataset_func_list = sorted(list(func_dicts[t].keys()))
            dataset_results = results[t]
            for i in range(len(dataset_func_list) + 1):
                if i != len(dataset_func_list):
                    tp, pred, true = sum([r[i] for r in dataset_results["tp"]]), sum([r[i] for r in dataset_results["pred"]]), sum([r[i] for r in dataset_results["true"]])
                else:
                    # 4 is for all functions
                    tp, pred, true = sum([r.sum() for r in dataset_results["tp"]]), sum([r.sum() for r in dataset_results["pred"]]), sum([r.sum() for r in dataset_results["true"]])
                # print(f"tp: {tp}, pred: {pred}, true: {true}")

                if local_rank == 0:
                    if i != len(dataset_func_list) and log_each:
                        print()
                        print({
                            f"data-{datasets[t]}_test-tp-{dataset_func_list[i]}": tp,
                            f"data-{datasets[t]}_test-pred-{dataset_func_list[i]}": pred,
                            f"data-{datasets[t]}_test-true-{dataset_func_list[i]}": true,
                            f"data-{datasets[t]}_test-precision-{dataset_func_list[i]}": tp / (pred + 1e-8),
                            f"data-{datasets[t]}_test-recall-{dataset_func_list[i]}": tp / (true + 1e-8),
                            f"data-{datasets[t]}_test-f1-{dataset_func_list[i]}": 2 * tp / (pred + true + 1e-8)
                        })
                        print()
                    elif i == len(dataset_func_list):
                        print()
                        print({
                            f"data-{datasets[t]}_test-tp": tp,
                            f"data-{datasets[t]}_test-pred": pred,
                            f"data-{datasets[t]}_test-true": true,
                            f"data-{datasets[t]}_test-precision": tp / (pred + 1e-8),
                            f"data-{datasets[t]}_test-recall": tp / (true + 1e-8),
                            f"data-{datasets[t]}_test-f1": 2 * tp / (pred + true + 1e-8)
                        })
                        print()
        # lr_scheduler.step()

        # save the parameters of func_embed every epoch
        save_dir = "-".join([d.replace("-", "_") for d in datasets])
        save_dir = f"checkpoints/{save_dir}{log_suffix}/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(funcmodel.func_embed.state_dict(), f"{save_dir}/epoch_{epoch}.pth")

        # results = [defaultdict(list) for _ in range(len(datasets))]


if __name__ == "__main__":
    # fire.Fire(main)
    MODEL = "meta-llama/Llama-3.2-1B"

    dataset1: str = "kamel"
    input_file1: str = "data/kamel/Llama-3.2-1B-kamel_id_train.json"
    lr: float = 1e-4
    num_epochs: int = 2
    only_functoken=False
    func_emb_file = None
    log_suffix="-multi-debug"
    # # log_each=False

    # dataset: str = "funcqa"
    # input_file: str = "data/funcqa/Llama-3.2-1B-funcqa_id_train.json"
    # lr: float = 1e-3
    # num_epochs: int = 10
    # only_functoken=False
    # func_emb_file = None
    # # log_prefix=""
    # # log_each=False

    dataset2: str = "gsm8k-xl"
    input_file2: str = "data/gsm8k-xl/Llama-3.2-1B_gsm8k_train.json"
    # lr: float = 3e-4        # 1e-4 worked for non-quantized version
    # num_epochs: int = 3
    # only_functoken=False
    # func_emb_file = None
    # log_prefix=""
    # log_each=False

    log_index = 200
    batch_save_index = 500

    print(f"dataset 1: [{dataset1}]")
    print(f"input_file 1: [{input_file1}]")
    print(f"dataset 2: [{dataset2}]")
    print(f"input_file 2: [{input_file2}]")
    print(f"lr: [{lr}]")
    print(f"num_epochs: [{num_epochs}]")
    print(f"only_functoken: [{only_functoken}]")
    print(f"log_index: [{log_index}]")
    print(f"batch_save_index: [{batch_save_index}]")

    datasets = [dataset1, dataset2]
    input_files = [input_file1, input_file2]

    main(
        model_uri = MODEL,
        datasets = datasets,
        input_files = input_files,
        lr = lr,
        num_epochs = num_epochs,
        only_functoken = only_functoken,
        log_each = False,
        log_suffix = log_suffix,
        func_emb_file = func_emb_file,
        log_index = log_index,
        batch_save_index = batch_save_index,
        debug = False,
    )
