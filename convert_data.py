# -*- coding: utf-8 -*-
# author: Mrinaal Dogra (mdogra@ucsd.edu)

import json
import os

from typing import Any, Dict, List, Set, Tuple, Union

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def convert_train_data(
        input_data: List[Dict[str, Any]],
        input_tokenizer,
        output_tokenizer,
        ignore_mismatches: bool = False,
        ignore_indices: List[int] = [],
        bos: bool = True,
        eos: bool = True
) -> List[Dict[str, Any]]:
    output_data = []
    for data_index, data in tqdm(enumerate(input_data), total=len(input_data)):
        if data_index in ignore_indices:
            continue
        input_text = data["text"]
        input_start = data["start_token_idx"]
        input_end = data["end_token_idx"]
        output_start, output_end = [], []

        input_tokens = input_tokenizer.encode(input_text, add_special_tokens=False)
        output_tokens = output_tokenizer.encode(input_text, add_special_tokens=False)
        if bos:
            input_tokens = [input_tokenizer.bos_token_id] + input_tokens
            output_tokens = [output_tokenizer.bos_token_id] + output_tokens
        if eos:
            input_tokens = input_tokens + [input_tokenizer.eos_token_id]
            output_tokens = output_tokens + [output_tokenizer.eos_token_id]

        # print("*"*50)
        output_token_index = 0
        is_data_valid = True
        for s, e in zip(input_start, input_end):
            prefix = input_tokenizer.decode(input_tokens[:s], skip_special_tokens=True)
            prefix_joined = ''.join(prefix.split())
            text = input_tokenizer.decode(input_tokens[:e], skip_special_tokens=True)
            text_joined = ''.join(text.split())

            output_s, output_e = len(output_tokens) + 1, len(output_tokens) + 1
            for i in range(output_token_index, len(output_tokens)):
                output_text = output_tokenizer.decode(output_tokens[:i], skip_special_tokens=True)
                output_text_joined = ''.join(output_text.split())

                # if 80 < i and i < 90:
                #     display(f"{i:3d} -> {output_text}")

                if output_text_joined == prefix_joined:
                    output_s = i
                elif output_text_joined == text_joined:
                    output_e = i
                    output_token_index = i
                    break

            # print(s, e, output_s, output_e)
            # display(len(text), text)
            # output_text = output_tokenizer.decode(output_tokens[:output_e], skip_special_tokens=True)
            # display(len(output_text), output_text)
            # print(f"[{input_tokenizer.decode(input_tokens[s:e], skip_special_tokens=True)}]")
            # print(f"[{output_tokenizer.decode(output_tokens[output_s:output_e], skip_special_tokens=True)}]")
            # print()

            if ignore_mismatches:
                if output_s >= len(output_tokens):
                    is_data_valid = False
                    continue
                if output_e >= len(output_tokens):
                    is_data_valid = False
                    continue
                if input_tokenizer.decode(input_tokens[s:e], skip_special_tokens=True).strip() != output_tokenizer.decode(output_tokens[output_s:output_e], skip_special_tokens=True).strip():
                    is_data_valid = False
                    continue

            assert output_s < len(output_tokens), print(f"{data_index}"
                + "\n"
                + f"{text}"
                + "\n"
                + f"{output_tokenizer.decode(output_tokens[:output_e], skip_special_tokens=True)}"
                + "\n"
                + f"[{input_tokenizer.decode(input_tokens[s:e], skip_special_tokens=True)}] "
                + f"[{output_tokenizer.decode(output_tokens[output_s:output_e], skip_special_tokens=True)}]"
            )
            assert output_e < len(output_tokens), print(f"{data_index}"
                + "\n"
                + f"{text}"
                + "\n"
                + f"{output_tokenizer.decode(output_tokens[:output_e], skip_special_tokens=True)}"
                + "\n"
                + f"[{input_tokenizer.decode(input_tokens[s:e], skip_special_tokens=True)}] "
                + f"[{output_tokenizer.decode(output_tokens[output_s:output_e], skip_special_tokens=True)}]"
            )

            assert input_tokenizer.decode(input_tokens[s:e], skip_special_tokens=True).strip() == output_tokenizer.decode(output_tokens[output_s:output_e], skip_special_tokens=True).strip(), print(""
                + f"{data_index}\n"
                + f"{input_tokenizer.decode(input_tokens[s:e], skip_special_tokens=True)}\n"
                + f"{output_tokenizer.decode(output_tokens[output_s:output_e], skip_special_tokens=True)}"
            )

            output_start.append(output_s)
            output_end.append(output_e)

        # # gsm8k
        # output_data.append({
        #     "text": input_text,
        #     "start_token_idx": output_start,
        #     "end_token_idx": output_end,
        #     "tar_eq": data["tar_eq"],
        #     "tar_number": data["tar_number"],
        # })

        if is_data_valid:
            odata = {k: v for k, v in data.items()}
            odata["text"] = input_text
            odata["start_token_idx"] = output_start
            odata["end_token_idx"] = output_end

            output_data.append(odata)

    return output_data


# DATASET_ROOT = "data/gsm8k-xl"
# DATASET_ROOT = "data/funcqa"
# DATASET_ROOT = "data/kamel"
DATASET_ROOT = "data/vh"

func_dict_file = os.path.join(DATASET_ROOT, "func_dict.json")
func_dict = {}
with open(func_dict_file, "r") as f:
    func_dict = json.load(f)

# input_file_path = os.path.join(DATASET_ROOT, "train.json")                      # gsm8k
# input_file_path = os.path.join(DATASET_ROOT, "train.json")                      # funcqa
# input_file_path = os.path.join(DATASET_ROOT, "kamel_id_train.json")             # Kamel supervised
# input_file_path = os.path.join(DATASET_ROOT, "train_clean.json")                # Kamel synthetic
input_file_path = os.path.join(DATASET_ROOT, "legal_train_v4_embedding.json")   # vh
input_data = {}
with open(input_file_path, "r") as f:
    input_data = json.load(f)

# l1_tokenizer = AutoTokenizer.from_pretrained("dfurman/llama-7b")
# l1_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
# l1_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-13b")
l1_tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-30b")

l3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# # For gsm8k-xl/train.json
# # For kamel/kamel_id_train.json
# output_data = convert_train_data(input_data, l1_tokenizer, l3_tokenizer)

# For kamel/train_clean.json
output_data = convert_train_data(input_data, l1_tokenizer, l3_tokenizer, ignore_mismatches=True)



# output_file = os.path.join(DATASET_ROOT, "Llama-3.2-1B_gsm8k_train.json")                   # gsm8k
# output_file = os.path.join(DATASET_ROOT, "Llama-3.2-1B_funcqa_train.json")                  # funcqa
# output_file = os.path.join(DATASET_ROOT, "Llama-3.2-1B-kamel_id_train.json")                # Kamel (supervised)
# output_file = os.path.join(DATASET_ROOT, "Llama-3.2-1B-kamel_train_clean.json")             # Kamel (synthetic)
output_file = os.path.join(DATASET_ROOT, "Llama-3.2-1B_vh_legal_train_v4_embedding.json")   # vh
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)
print(f"Output saved in [{output_file}]")
