# CSE291A Project - ToolkenGPT
This project is forked from the [original project repo](https://github.com/Ber666/ToolkenGPT)

The goal of the project was to explore the effectiveness of ToolkenGPT framework with more recent and smaller language models like Llama-3.2. In this endeavor, the code required significant modifications in order to use Hugging Face's Transformer library-based model implementations. This allowed for a more flexible framework that can be used to explore different models with ease.

Moreover, the project also aimed at exploring multi-task learning-based strategies and how the ToolkenGPT framework performs when more than one task (like numerical calculation and knowledge-based QA) are trained simultaneously.

Major Files updated from the original source code:
- `train_llama.py`: This file defines the training pipeline, and was changed to support Hugging Face Transformer library-based implementations
- `llama/model.py`: This file defines the ToolkenGPT framework model called `FunctionLM` and its training and inference methods. This was also updated to support Hugging Face Transformer library-based implementations
- `inference_llama.py`: This file defines the inference pipeline, and was changed to support the Hugging Face Transformer library-based implementations

Files newly added
- `convert_data.py`: Python module used to re-annotate the original dataset with Llama-3.2 tokenizer.
- `eval_gsm8k_funcqa.py`: Python module inspired from `evaluation/eval_funcqa.ipynb` file to evaluate the inference results of GSM8K-XL and FuncQA datasets.
- `eval_kamel.py`: Python module inspired from `evaluation/eval_kamel.ipynb` file to evaluate the inference results of KAMEL datasets.
- `train_llama_multitask.py`: Python module that defines the training pipeline for the Multitask ToolkenGPT framework
- `llama/multitask_model.py`: Python module that defines the Multitask model called `MultiTaskFunctionLM` and its training and inference methods. This model implementation still has some bugs, so it can only work for the first task from the list of multiple tasks on which the model was trained.
- `inference_llama_multitask.py`: Python module that helps in generating inference results for the multi-task trained model.
- `inference_modes_multitask.py`: Python module inspired from `inference_modes.py` that defines helper methods for conducting inference in the multi-task paradigm.


# Content from original project
**Source code for [ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings](https://arxiv.org/abs/2305.11554)**

[NeurIPS 2023 (oral)](https://nips.cc/Conferences/2023) | [Best Paper Award at SoCalNLP 2023](https://socalnlp.github.io/symp23/index.html)

![Figure](assets/image.png)

## Preparation
+ Our experiments are conducted with LLaMA-13B/33B, which takes at least 2/4 GPUs of 24GB memory each.
+ Acquire the checkpoints of LLaMA from MetaAI and install all required packages. Please refer to [LLaMA official repo](https://github.com/facebookresearch/llama).
+ Download the data from [here](https://drive.google.com/file/d/13Sj7uIsyqWXoTh1ejWUviTzeQSES2Omd/view?usp=sharing) (all datasets uploaded)
+ (For VirtualHome) Please download the data following the instructions [here](virtualhome/README.md).
    > A side note: the folder `virtualhome` is from its [official repo](https://github.com/xavierpuigf/virtualhome), but we fixed some small bugs in the evolving graph.

## GSM8K-XL

### Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1200 train_llama.py --ckpt_dir $LLAMA_CKPTS/30B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --input_file data/gsm8k-xl/train.json --lr 1e-3 --num_epochs 10
```

### Inference

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1250 inference_llama.py --ckpt_dir $LLAMA_CKPTS/30B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --mode func_embedding --dataset gsm8k-xl  --func_load_path checkpoints/gsm8k-xl/epoch_3.pth --logits_bias 3.0
```

## FuncQA

### Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1200 train_llama.py --ckpt_dir $PATH_TO_LLAMA/30B --tokenizer_path $PATH_TO_LLAMA/tokenizer.model --input_file data/funcqa/train.json --lr 1e-4 --num_epochs 10
```

### Inference (1-hop)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1250 inference_llama.py --ckpt_dir $LLAMA_CKPTS/30B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --mode func_embedding --dataset funcqa_oh --func_load_path checkpoints/funcqa/epoch_7.pth --logits_bias 2.7
```

### Inference (MultiHop)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 --master_port 1250 inference_llama.py --ckpt_dir $LLAMA_CKPTS/30B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --mode func_embedding --dataset funcqa_mh --func_load_path checkpoints/funcqa/epoch_7.pth --logits_bias 4.0
```

## VirtualHome

### Training
```bash
python -m torch.distributed.run --nproc_per_node 2 --master_port 3001 train_llama.py --ckpt_dir $LLAMA_CKPTS/13B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --dataset vh --input_file data/vh/legal_train_v4_embedding.json --only_functoken True --num_epochs 10
```


### Inference

```bash
CUDA_VISIBLE_DEVICES=3,5 python -m torch.distributed.run --nproc_per_node 2 inference_llama.py --ckpt_dir $LLAMA_CKPTS/13B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --mode vh_embedding_inference --dataset vh --func_load_path checkpoints/vh/epoch_7.pth --logits_bias 10.0
```

### Evaluation

See `evaluation/eval_vh.ipynb`

## KAMEL
### Train
+ synthetic data
```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port 3002 train_llama.py --ckpt_dir $LLAMA_CKPTS/13B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --dataset kamel --input_file data/kamel/train_clean.json --only_functoken False ---log_every 500 --num_epochs 10
```


+ supervised data
```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 --master_port 3002 train_llama.py --ckpt_dir $LLAMA_CKPTS/13B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --dataset kamel --input_file data/kamel/kamel_id_train.json --only_functoken False ---log_every 500 --num_epochs 10
```

### Inference

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node 2 inference_llama.py --ckpt_dir $LLAMA_CKPTS/13B --tokenizer_path $LLAMA_CKPTS/tokenizer.model --mode kamel_embedding_inference --dataset kamel_30 --func_load_path checkpoints/kamel/epoch_4.pth --logits_bias 10
```

### Evaluation

See `evaluation/eval_kamel.ipynb`
