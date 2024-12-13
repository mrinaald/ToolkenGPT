import time
import re
from funchub.math import *

def func_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top=5):
    cur_generation = ""
    cur_generation_with_func = ""
    start_length = []
    end_length = []
    logs = []
    funcmodel.inference_mode = "func_embedding"
    func_map = list(funcmodel.func_dict.keys())
    status = "success"
    try:
        results = []
        func_calls = []
        whole_iter_s = time.time()
        while True and status == "success":
            iter_s = time.time()
            prompt = templates["general"].replace("[QUESTION]", question) + cur_generation
            ############################################################
            # CAUTION:
            # stop_token=[13] is for Llama-1 tokenizer
            # For using any other tokenizer, try to use the following:
            # ```
            # [l3_tokenizer.encode(l1_tokenizer.decode([t]), add_special_tokens=False)[0] for t in tokens]
            # ```
            # The below value is for Llama-3.2 tokenizer
            ############################################################
            # results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[13], return_top=return_top)
            g1_s = time.time()
            results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[198], return_top=return_top)
            g1_e = time.time()
            print(f"G1 time: {g1_e - g1_s:.2f} s", flush=True)

            if return_top > 0:
                results, token_log = results
                logs.append(token_log)
            endflag = True
            current_token = 0
            record_tokens = token_log[-1]
            cur_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "")
            for op in func_map:
                if cur_generation.endswith(op+"("):
                    endflag = False
                    if start_length and end_length:
                        bias = 0
                        # copy the current generation to cur_generation_with_func
                        cur_generation_with_func = cur_generation
                        for i in range(len(start_length)):
                            cur_generation_with_func = cur_generation_with_func[:start_length[i]+bias] +func_calls[i] + cur_generation_with_func[end_length[i]+bias:]
                            bias += len(func_calls[i]) - (end_length[i] - start_length[i])
                    else:
                        cur_generation_with_func = cur_generation
                    prompt = templates[op].replace("[QUESTION]", question) + cur_generation_with_func
                    len_prompt = len(prompt)
                    funcmodel.inference_mode = "baseline"

                    ############################################################
                    # CAUTION:
                    # stop_token=[29897, 3892] is for Llama-1 tokenizer
                    # For using any other tokenizer, try to use the following:
                    # ```
                    # [l3_tokenizer.encode(l1_tokenizer.decode([t]), add_special_tokens=False)[0] for t in tokens]
                    # ```
                    # The below value is for Llama-3.2 tokenizer
                    ############################################################
                    # results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[29897, 3892], return_top=return_top)
                    g2_s = time.time()
                    results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[8, 11992], return_top=return_top)
                    g2_e = time.time()
                    print(f"G2 time: {g2_e - g2_s:.2f} s", flush=True)

                    funcmodel.inference_mode = "func_embedding"
                    if return_top > 0:
                        results, token_log = results
                        logs.append(token_log)
                    generated = results[0][len_prompt:]
                    cur_generation += generated
                    args = cur_generation.split(op)[-1].replace("=", "").replace(">", "").replace("((", "(").replace("))", ")")
                    # remove any $ in the args
                    args = args.replace("$", "")
                    # remove , in the args
                    if ", " in args:
                        args = args.replace(", ", ";").replace(",", "").replace(";", ", ")

                    args = args.replace(" ", "")
                    if "(" not in args or ")" not in args:
                        # raise Exception("invalid args")
                        status = f'Exception("invalid args: [{args}]")'
                        print(status)
                        break
                    # handle %
                    if '%' in args:
                        temp = args.split("(")[1].split(")")[0].split(",")
                        for arg_i, arg in enumerate(temp):
                            # if have percentage, convert to decimal
                            if "%" in arg:
                                arg = arg.replace("%", "").strip()
                                arg = str(float(arg) / 100)
                            temp[arg_i] = arg
                        args = f"({', '.join(temp)})"
                    try:
                        res = eval(f"{op[1:-1]}_{args}")
                        func_calls.append(f"{op}{args} = {res}")
                        start_length.append(len(cur_generation.split(op)[0]))
                        cur_generation = cur_generation.split(op)[0] + str(res)
                        end_length.append(len(cur_generation))

                        # only generate the next token
                        # disable all the numbers
                        prompt = templates["general"].replace("[QUESTION]", question) + cur_generation

                        ############################################################
                        # CAUTION:
                        # stop_token=[13] is for Llama-1 tokenizer
                        # disable_token = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929] is for Llama-1 tokenizer
                        #
                        # For using any other tokenizer, try to use the following:
                        # ```
                        # [l3_tokenizer.encode(l1_tokenizer.decode([t]), add_special_tokens=False)[0] for t in tokens]
                        # ```
                        # The below values is for Llama-3.2 tokenizer
                        ############################################################
                        # results = funcmodel.generate([prompt], max_gen_len=1, temperature=temperature, top_p=top_p, stop_token=[13], return_top=return_top, disable_token = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]) # disable all the numbers: 0-9
                        g3_s = time.time()
                        results = funcmodel.generate([prompt], max_gen_len=1, temperature=temperature, top_p=top_p, stop_token=[198], return_top=return_top, disable_token = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]) # disable all the numbers: 0-9
                        g3_e = time.time()
                        print(f"G3 time: {g3_e - g3_s:.2f} s", flush=True)

                        if return_top > 0:
                            results, token_log = results
                            logs.append(token_log)
                        cur_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "")
                    except:
                        # backtrace
                        current_token += 1
                        # decode_token = lambda x: funcmodel.tokenizer.decode(x) if x < 32000 else func_map[x - 32000]
                        decode_token = lambda x: funcmodel.tokenizer.decode(x) if x < funcmodel.toolken_offset else func_map[x - funcmodel.toolken_offset]
                        cur_generation = cur_generation.split(op)[0] + decode_token(record_tokens[1][current_token][0])
                    break
            iter_e = time.time()
            print(f"Iter time: {iter_e - iter_s:.2f} s", flush=True)
            if endflag:
                break

        whole_iter_e = time.time()
        print(f"whole iter time: {whole_iter_e - whole_iter_s:.2f} s", flush=True)
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            # "status": "success"
            "status": status,
        }

    except Exception as e:
        raise e
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": str(e)
        }
    return log


def vh_embedding_inference(case_idx, question, funcmodel, temperature, top_p, max_func_call):
    funcmodel.inference_mode = "func_embedding"
    inputs = question[0]
    disable_funcs = question[1]
    last_func = []
    for _ in range(max_func_call):
        inputs = funcmodel.generate([inputs], max_gen_len=1, temperature=temperature, top_p=top_p,return_top=0, disable_func=disable_funcs + last_func, no_left_parens=True)[0]

        if inputs.endswith(">"):
            inputs = inputs.replace("]<", "] <")
            inputs += '\n'
            last_func = [] if "[WALK]" in inputs.split("\n")[-2] else re.findall(r"\[.*?\]", inputs.split("\n")[-2])
            print("last func", last_func)
        if "[END]" in inputs.split("Plan:")[-1]:
            break


    log = {
    "case_idx": case_idx,
    "question": question[0],
    "func_calls": inputs.replace(question[0], "").strip().split("\n"),
    "generation": inputs.replace("\n", "\\n").strip(),
    # no need to return logs
    # "token_log": logs,
    "status": "success"
    }
    return log


def kamel_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, max_func_call):

    funcmodel.inference_mode = "func_embedding"
    cur_generation = ""
    if "funcgeneral" not in templates:
        templates["funcgeneral"] = templates["general"]
    try:
        results = []
        func_calls = []
        while True:
            if max_func_call == 0:
                break
            prompt = templates["funcgeneral"].replace("[QUESTION]", question) + cur_generation

            ############################################################
            # CAUTION:
            # stop_token=[13] is for Llama-1 tokenizer
            # For using any other tokenizer, try to use the following:
            # ```
            # [l3_tokenizer.encode(l1_tokenizer.decode([t]), add_special_tokens=False)[0] for t in tokens]
            # ```
            # The below value is for Llama-3.2 tokenizer
            ############################################################
            # results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[13])
            results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[198])

            max_func_call -= 1

            cur_generation = results[0].replace(templates["funcgeneral"].replace("[QUESTION]", question), "")
            # one function token is enough
            break
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": "success"
        }
        # f.write(json.dumps(log) + "\n")

    except Exception as e:
        # if local_rank == 0:
        raise e
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": str(e)
        }
    return log
