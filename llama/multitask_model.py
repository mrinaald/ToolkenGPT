# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import math
import time
import torch
from torch import nn
import torch.nn.functional as F
import copy
import random
import json
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)
from transformers import AutoTokenizer, AutoModel
import re

@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_local_heads = args.n_heads // fs_init.get_model_parallel_world_size()
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()
        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_local_heads, self.head_dim)
        ).cuda()

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim, hidden_dim=4 * args.dim, multiple_of=args.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    # @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)  # (bsz, partial_seqlen, dim)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h[:, -1, :])  # only compute last logits
        return output.float(), h

class MultiTaskFunctionLM(nn.Module):
    def __init__(self, base_model, tokenizer, func_dicts, device="cpu", load_path=None, inference_mode="func_embedding"):
        super().__init__()
        self.inference_mode = inference_mode
        self.model = base_model
        self.tokenizer = tokenizer
        self.func_dicts = func_dicts
        self.func_lists = [{v: k for k, v in func_dict.items()} for func_dict in self.func_dicts]
        self.func_embed_length = sum([len(func_dict) for func_dict in self.func_dicts])
        self.device = device
        self.precomputed_hidden_state = {}
        self.precomputed_token_logits = {}
        # self.toolken_offset = tokenizer.vocab_size            # True for Llama-1, not for Llama-3.2
        offset = max(max([id for _, id in tokenizer.vocab.items()]) + 1, tokenizer.vocab_size)
        self.toolken_offsets = []
        for i in range(len(self.func_dicts)):
            self.toolken_offsets.append(offset)
            offset += len(self.func_dicts[i])

        for i in range(len(self.toolken_offsets)):
            print(f"Toolken {i} offset: {self.toolken_offsets[i]}")

        # self.func_embed = ColumnParallelLinear(
        #     base_model.params.dim, len(func_list), bias=False, init_method=lambda x: x
        # )

        # # For Llama-1
        # self.func_embed = nn.Linear(base_model.params.dim, len(func_dict), bias=False).to(device)
        # For Llama-3.2
        self.func_embed = nn.Linear(base_model.config.hidden_size, self.func_embed_length, bias=False).to(device)
        # self.func_embed = nn.Linear(base_model.config.hidden_size, len(func_dict), bias=False, dtype=self.model.dtype).to(device)

        # self.layer_norm = nn.LayerNorm(base_model.config.hidden_size).to(device)

        print(f"Trying to load... [{load_path}]")
        if load_path is not None and load_path != "None": # load func_embed weights
            embedding = torch.load(load_path)
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.to(device)
                embedding = {"weight": embedding}

            # TODO: Fix this truncation logic, as currently it keeps the toolkens only from the beginning, and the truncated dataset might not be the first one
            # # truncate the embedding if necessary
            if embedding["weight"].shape[0] > len(self.func_dicts[0]):
                # assert False, "Truncation is supported only for 0-th dataset. If you are evaluating the 0-th dataset, then comment this assertion"
                self.func_embed = nn.Linear(base_model.config.hidden_size, len(self.func_dicts[0]), bias=False).to(device)
                print(f"Truncated the function embedding from {embedding['weight'].shape[0]} to {len(self.func_dicts[0])}")
                embedding["weight"] = embedding["weight"][:len(self.func_dicts[0])]

            self.func_embed.load_state_dict(embedding)
            print(f"Function Embeddings loaded from file [{load_path}]")

        # set the basemodel to eval mode and freeze the weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.logits_bias = 0

        # self.model = torch.compile(self.model)

    def set_bias(self, logits_bias):
        self.logits_bias = logits_bias

    def get_loss_for_type(self, raw_input, toolken_type, only_functoken=False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        assert 0 <= toolken_type and toolken_type < len(self.func_dicts), f"Unknown toolken type: {toolken_type}"
        func_dict = self.func_dicts[toolken_type]
        toolken_offset = self.toolken_offsets[toolken_type]
        # assert len(raw_input) == 1
        # raw_input = raw_input[0]

        # inputs: starts with <bos>, ends without <eos>, (bsz, seqlen)
        # labels: starts without <bos>, ends with <eos>, (bsz, seqlen)
        with torch.no_grad():
            # prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=True) for x in raw_inputs]

            # # For Llama-1
            # raw_input_ids = torch.tensor(self.tokenizer.encode(raw_inputs["text"], bos=True, eos=True))[:]
            # labels = torch.tensor(self.tokenizer.encode(raw_inputs["text"], bos=True, eos=True))[:]
            # For Llama-3.2
            raw_input_ids = torch.tensor([self.tokenizer.bos_token_id] + self.tokenizer.encode(raw_input["text"], add_special_tokens=False) + [self.tokenizer.eos_token_id])[:]
            labels = torch.tensor([self.tokenizer.bos_token_id] + self.tokenizer.encode(raw_input["text"], add_special_tokens=False) + [self.tokenizer.eos_token_id])[:]

            if "tar_eq" not in raw_input:
                raw_input["tar_eq"] = ["<" + raw_input["api"] + ">"]

            for s, t, eq in zip(raw_input["start_token_idx"], raw_input["end_token_idx"], raw_input["tar_eq"]):

                # for different data formats
                if "[" in eq:
                    op = re.search(r"(\[.*?\])", eq).group(1)
                elif "<" in eq:
                    op = re.search(r"(<.*?>)", eq).group(1)
                    # print(op)

                if op not in func_dict:
                    op = op[1:-1]
                labels[s] = func_dict[op] + toolken_offset
                labels[s+1: t] = -100

            # labels = labels[1:]
            if only_functoken:
                labels[labels < toolken_offset] = -100
            inputs = raw_input_ids[:-1].expand(1, -1).to(self.device)
            labels = labels[1:].expand(1, -1).to(self.device)

            # if raw_inputs["text"] not in self.precomputed_hidden_state:

            # # original
            # last_logits, h = self.model(inputs, 0) # h: (bsz, seqlen, dim)
            # token_logits = self.model.output(h) # (bsz, seqlen, vocab_size)
            # # print(h.device)
            # With AutoModelForCausalLM
            outputs = self.model(inputs, output_hidden_states=True)
            h = outputs.hidden_states[-1]
            token_logits = outputs.logits

            # self.precomputed_hidden_state[raw_inputs["text"]] = h
            # self.precomputed_token_logits[raw_inputs["text"]] = token_logits

            # else:
                # h = self.precomputed_hidden_state[raw_inputs["text"]]
                # token_logits = self.precomputed_token_logits[raw_inputs["text"]]

        # h = self.layer_norm(h)
        # func_logits = self.func_embed(h.float()) # (bsz, seqlen, len(func_list))
        func_logits = self.func_embed(h.to(self.func_embed.weight.dtype)) # (bsz, seqlen, len(func_list))

        concat_logits = torch.cat([token_logits, func_logits], dim=-1) # (bsz, seqlen, vocab_size + len(func_list))
        # loss = F.cross_entropy(concat_logits.view(-1, concat_logits.shape[-1]), labels.view(-1), ignore_index=-100)
        loss = F.cross_entropy(concat_logits.view(-1, concat_logits.shape[-1]).float(), labels.view(-1), ignore_index=-100)
        # check p, r, f1 for each function
        pred = torch.argmax(concat_logits, dim=-1) # (bsz, seqlen)
        pred = pred.view(-1)
        labels = labels.view(-1)

        label_funcs = [labels == func_dict[op] + toolken_offset for op in func_dict.keys()]
        pred_funcs = [pred == func_dict[op] + toolken_offset for op in func_dict.keys()]
        label_funcs = torch.stack(label_funcs, dim=0)
        pred_funcs = torch.stack(pred_funcs, dim=0)

        tp = torch.sum(label_funcs * pred_funcs, dim=-1).detach().cpu().numpy()
        pred_funcs = torch.sum(pred_funcs, dim=-1).detach().cpu().numpy()
        true = torch.sum(label_funcs, dim=-1).detach().cpu().numpy()
        results = {
            "tp": tp,               # len == num toolkens
            "pred": pred_funcs,     # len == num toolkens
            "true": true            # len == num toolkens
        }

        return loss, results

    def get_loss(self, all_raw_inputs, only_functoken=False) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        global_loss = []
        global_results = []
        for toolken_type, raw_inputs in enumerate(all_raw_inputs):
            loss = 0
            results = {}
            for raw_input in raw_inputs:
                sample_loss, sample_results = self.get_loss_for_type(raw_input, toolken_type, only_functoken=only_functoken)

                loss += sample_loss

                # Update results
                for k, v in sample_results.items():
                    if k not in results:
                        results[k] = v
                    else:
                        results[k] += v

            global_loss.append(loss / len(raw_inputs))
            global_results.append(results)

        return global_loss, global_results

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        toolken_type: int,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_token: List[int] = [], # 29897: ), 3892: )=
        return_top: int = 0,
        disable_func: List[str] = [],
        disable_token: List[int] = [], # 29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        no_left_parens: bool = False,

        objs: List[str] = [],
    ) -> List[str]:

        # print("\n" + "*"*50)

        assert toolken_type == 0, "The current implementation is unable to load saved model embeddings in case the first datatype is truncated KAMEL. If you are not using KAMEL as first dataset, this assertion can be removed"
        func_dict = self.func_dicts[toolken_type]
        toolken_offset = self.toolken_offsets[toolken_type]
        func_list = self.func_lists[toolken_type]

        bsz = len(prompts)
        # print("objs", objs)

        # # Llama-1
        # obj_encodings = [self.tokenizer.encode("<"+obj+">", bos=False, eos=False)[1:-1] for obj in objs]
        # Llama-3.2
        obj_encodings = [self.tokenizer.encode("<"+obj+">", add_special_tokens=False)[1:-1] for obj in objs]
        # print("obj encoding", obj_encodings)
        assert bsz == 1
        if self.device == "cuda":
            stop_token_substr = [torch.tensor(x).cuda().long() for x in stop_token if isinstance(x, list)]
        else:
            stop_token_substr = [torch.tensor(x).long() for x in stop_token if isinstance(x, list)]
        stop_token_single = [x for x in stop_token if isinstance(x, int)]
        # print(f"stop token: {stop_token} | {stop_token_substr} | {stop_token_single}")

        funcs = [func_list[v] for v in range(len(func_dict))]

        # tokenize all the func in func_list
        # # Llama-1
        # func_tokens = [self.tokenizer.encode(x[1:-1], bos=False, eos=False) for x in func_list]
        # Llama-3.2
        func_tokens = [self.tokenizer.encode(x[1:-1], add_special_tokens=False) for x in funcs]

        generation_log = [] # (token, [(token, logits, prob)])
        # params = self.model.params
        # assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)     # Batch size not available with Llama-3.2

        # # Llama-1
        # prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        # Llama-3.2
        prompt_tokens = [[self.tokenizer.bos_token_id] + self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        # total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)
        # print(f"min_prompt_size: {min_prompt_size} | max_prompt_size: {max_prompt_size} | max_gen_len: {max_gen_len}")
        total_len = min(self.model.config.max_position_embeddings, max_gen_len + max_prompt_size)

        # # Llama-1
        # tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        # for k, t in enumerate(prompt_tokens):
        #     tokens[k, : len(t)] = torch.tensor(t).long()
        # input_text_mask = tokens != self.tokenizer.pad_id

        # # Llama-3.2
        pad_id = self.toolken_offsets[0] - 1        # Using last reserved special token as padding. Adding a separate special token in tokenizer updates the vocab size, thus affecting the toolken indices

        if self.device == "cuda":
            tokens = torch.full((bsz, total_len), pad_id).cuda().long()
        else:
            tokens = torch.full((bsz, total_len), pad_id).long()

        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()

        input_text_mask = tokens != pad_id
        start_pos = min_prompt_size
        past_key_values = None
        prev_pos = 0
        hs = []

        # print(f"start_pos: {start_pos} | total_len: {total_len}")
        loop_s = time.time()
        for cur_pos_idx, cur_pos in enumerate(range(start_pos, total_len)):
            iter_s = time.time()
            """Test this
            # Forward pass
            outputs = model(input_ids, output_hidden_states=True)

            # Get the last hidden state
            last_hidden_state = outputs.hidden_states[-1]

            # Get the logits for the last token
            logits_token = outputs.logits[:, -1, :].float()
            """
            # # Llama-1
            # _, h = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            # logits_token = self.model.output(h[:, -1, :]).float() # (bsz, vocab_size)
            # logits_func = self.func_embed(h[:, -1, :].float()) # (bsz, len(func_list))

            # # Llama-3.2
            model_s = time.time()
            # outputs = self.model(tokens[:, :cur_pos], output_hidden_states=True)

            # # Llama-3.2 with caching
            # outputs = self.model(tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=past_key_values, output_hidden_states=True)
            outputs = self.model(tokens[:, prev_pos:cur_pos], attention_mask=input_text_mask, use_cache=True, past_key_values=past_key_values, output_hidden_states=True)
            model_e = time.time()
            # print(f"Model time: {model_e - model_s:.3f} s")

            h = outputs.hidden_states[-1]
            past_key_values = outputs.past_key_values
            logits_token = outputs.logits[:, -1, :].float()
            logits_func = self.func_embed(h[:, -1, :].to(self.func_embed.weight.dtype))


            if self.inference_mode != "func_embedding":
                logits_func = torch.zeros_like(logits_func) - 1e5

            if len(disable_token) > 0:
                logits_token[:, disable_token] = -1e5

            # topk: (bsz, 3)
            # print("after-topk", topk[1][0], [self.tokenizer.decode([x]) for x in topk[1][0].tolist()])

            for i, func in enumerate(disable_func):
                func_id = func_dict[func]
                logits_func[:, func_id] = -1e5

            logits_func += self.logits_bias
            logits = torch.cat([logits_token, logits_func], dim=-1)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if the prompt is ended
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            # TODO: Update input_text_mask
            # input_text_mask = torch.cat([input_text_mask, torch.ones((1, 1)).bool().to(self.device)], dim=-1)
            input_text_mask[:, cur_pos] = True

            if return_top > 0:
                generation_log.append(
                    (next_token[0].item(), [(i.item(), logits[0, i.item()].item()) for i in torch.argsort(logits[0, :], descending=True)[:return_top]])
                )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

            iter_e = time.time()
            # print(f"Iter time: {iter_e - iter_s:.3f} s")
            if next_token[0] >= toolken_offset or next_token[0] in stop_token_single:
                # print("breaking!!")
                break

            if any([torch.equal(tokens[0, cur_pos - len(substr) + 1: cur_pos + 1], substr) for substr in stop_token_substr]):
                break
        loop_e = time.time()
        # print(f"Loop time {loop_e - loop_s:.3f} s")

        # print(f"cur_pos: {cur_pos}", flush=True)
        decoded = []
        decode_s = time.time()
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # w = len(t)
            # cut to eos tok if any
            try:
                # # Llama-1
                # t = t[: t.index(self.tokenizer.eos_id)]
                # # Llama-3.2
                # x = t.index(self.tokenizer.eos_token_id)
                # y = [i for i, id in enumerate(t) if id == self.tokenizer.eos_token_id]
                t = t[: t.index(self.tokenizer.eos_token_id)]
                # z = len(t)
            except ValueError:
                # x = 0
                # y = []
                # z = 0
                pass

            # print(start_pos, max_gen_len, max_prompt_size, total_len, "|", cur_pos, i, w, x, y, z, len(t))
            # The following update is required in case EOS token comes before cur_pos
            if cur_pos >= len(t):
                cur_pos = len(t) - 1
            if t[cur_pos] >= toolken_offset:
                if no_left_parens:
                    decoded.append(self.tokenizer.decode(t[:cur_pos]) + func_list[t[cur_pos] - toolken_offset])
                else:
                    if "<" in func_list[0]:
                        decoded.append(self.tokenizer.decode(t[:cur_pos]) + func_list[t[cur_pos] - toolken_offset] + "(")
                    elif "[" in self.func_list[0]:
                        decoded.append(self.tokenizer.decode(t[:cur_pos]) + func_list[t[cur_pos] - toolken_offset] + " <")
                    else:
                        raise NotImplementedError
            else:
                decoded.append(self.tokenizer.decode(t[:cur_pos + 1]))
        decode_e = time.time()
        # print(f"Decode time: {decode_e - decode_s:.3f} s")

        # print("*"*50)
        if return_top > 0:
            return decoded, generation_log
        else:
            return decoded

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)

    return next_token