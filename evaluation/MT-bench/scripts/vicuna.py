import argparse
import json
import os
import time
import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm
import sys
sys.path.append('../../')#import CTC-drafter
from model.model import DraftModel
from model.kv_cache import initialize_past_key_values
from model.utils import *
from model.choices import *

use_safetensor_weight = False

def CTC_transform(candidates,retrieve_index,attn_mask,position_id,blank=0):
    prev = candidates[:,2:]
    rear = candidates[:,1:-1]
    prev_retrieve = retrieve_index[:,2:]
    zero_indice = torch.nonzero(torch.logical_and(prev == rear, prev_retrieve != -1), as_tuple=False)
    if torch.numel(zero_indice) == 0:# no modification for the attention map is needed
        return candidates,retrieve_index,attn_mask,position_id
    zero_indice[:,1] += 2 
    candidates[zero_indice[:, 0], zero_indice[:, 1]] = 0 #mask duplicate token
    need_to_mask = retrieve_index[zero_indice[:, 0], zero_indice[:, 1]]
    
    retrieve_index[zero_indice[:, 0], zero_indice[:, 1]] = -1
    
    need_to_mask = torch.unique(need_to_mask)
    need_to_mask = need_to_mask[need_to_mask != -1]
    
    attn_mask[:,:,need_to_mask,:] = 0
    attn_mask[:,:,:,need_to_mask] = 0
    
    non_zero_counts = (candidates != 0).sum(dim=1)
    non_zero_indices = torch.nonzero(candidates)
    start = 0
    for i in range(candidates.size(0)):
        if non_zero_counts[i] == candidates.size(1):#no zero
            start += non_zero_counts[i]
            continue

        non_zero_line = non_zero_indices[start:start+non_zero_counts[i]][:,1] # non-zero elements index
        candidates[i, :non_zero_counts[i]] = candidates[i, non_zero_line]
        candidates[i, non_zero_counts[i]:] = 0

        retrieve_index[i, :non_zero_counts[i]] = retrieve_index[i, non_zero_line]  # rearrange elements 
        retrieve_index[i, non_zero_counts[i]:] = -1
        
        start += non_zero_counts[i]
    # for i in range(candidates.size(0)):
    #     non_zero_indices = torch.nonzero(candidates[i, :]).squeeze(1)  # non-zero elements index
    #     zero_indices = torch.nonzero(candidates[i, :] == 0).squeeze(1)  
    #     sorted_indices = torch.cat((non_zero_indices, zero_indices))  
    #     candidates[i, :] = candidates[i, sorted_indices]  
    #     retrieve_index[i, :] = retrieve_index[i, sorted_indices]  # rearrange elements 
        
    return candidates,retrieve_index,attn_mask,position_id


def draft_forward(input_ids, model, tokenizer, tree_choices, max_steps=512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    # Avoid modifying the input_ids in-place
    input_ids = input_ids.clone()
    model.layer.reset_kv()

    if hasattr(model, "tree_choices") and model.tree_choices == tree_choices:
        tree_buffers = model.tree_buffers
    else:
        tree_buffers = generate_medusa_buffers(
            tree_choices, device=model.base_model.model.layers[-1].self_attn.q_proj.weight.device
        )
        tree_buffers["retrieve_indices_head"] = tree_buffers["retrieve_indices"].to(
            model.base_model.lm_head.weight.device)
    model.tree_buffers = tree_buffers
    model.tree_choices = tree_choices

    # Initialize the past key and value states
    if hasattr(model, "past_key_values"):
        past_key_values = model.past_key_values
        past_key_values_data = model.past_key_values_data
        current_length_data = model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(model.base_model)
        model.past_key_values = past_key_values
        model.past_key_values_data = past_key_values_data
        model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    reset_tree_mode(model)
    tree_logits, logits, hidden_state, sample_token = initialize_tree(
        input_ids, model, tree_buffers["medusa_attn_mask"], past_key_values
    )
    new_token = 0

    ##store the original attention map
    base_retrieve_indices = tree_buffers["retrieve_indices"].clone()
    base_medusa_attn_mask = tree_buffers["medusa_attn_mask"].clone()
    base_medusa_position_ids = tree_buffers["medusa_position_ids"].clone()
    
    for idx in range(max_steps):
        candidates, cart_candidates_prob, tree_candidates = generate_candidates(
            tree_logits,
            tree_buffers["tree_indices"],
            tree_buffers["retrieve_indices"],
            sample_token,
        )
        candidates,tree_buffers["retrieve_indices"],tree_buffers["medusa_attn_mask"],tree_buffers["medusa_position_ids"] = CTC_transform(
            candidates,
            tree_buffers["retrieve_indices"],
            tree_buffers["medusa_attn_mask"],
            tree_buffers["medusa_position_ids"],
        )
        logits, hidden_state_new, outputs = tree_decoding(
            model,
            tree_candidates,
            past_key_values,
            tree_buffers["medusa_position_ids"],
            input_ids,
            tree_buffers["retrieve_indices_head"],
            tree_buffers["medusa_attn_mask"]
        )
        tree_buffers["p_indices"] = None
        tree_buffers["b_indices"] = None
        best_candidate, accept_length, sample_p = evaluate_posterior(
            logits, candidates, cart_candidates_prob, tree_logits[2], tree_buffers["p_indices"],
            tree_candidates, tree_buffers["b_indices"]
        )
        input_ids, tree_logits, new_token, hidden_state, sample_token = update_inference_inputs(
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            tree_buffers["retrieve_indices"],
            logits,
            tree_logits,
            new_token,
            past_key_values_data,
            current_length_data,
            model,
            hidden_state,
            hidden_state_new,
            sample_p
        )
        tree_buffers["retrieve_indices"] = copy.deepcopy(base_retrieve_indices)
        tree_buffers["medusa_attn_mask"] = copy.deepcopy(base_medusa_attn_mask)
        tree_buffers["medusa_position_ids"] = copy.deepcopy(base_medusa_position_ids)          
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
        if input_ids.shape[1] > 1960:
            break
    return input_ids, new_token, idx


def run_eval(
        base_model_path,
        draft_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        num_gpus_total,
        max_gpu_memory,
        temperature,
        tree_choices,
):
    questions = load_questions(question_file, question_begin, question_end)

    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)  # // 2
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                draft_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                temperature,
                tree_choices,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        draft_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        num_gpus_per_model,
        max_gpu_memory,
        temperature,
        tree_choices,
):

    model = DraftModel.from_pretrained(
        base_model_path=base_model_path,
        draft_model_path=draft_model_path,
        use_safetensor_weight=use_safetensor_weight,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    tokenizer = model.get_tokenizer()
    
    model.eval()
    print('Check model training state:', model.training)

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    print('CUDA VISIBLE DEVICES:', cuda_visible_devices)

    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        conv = get_conversation_template("vicuna")
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer([prompt]).input_ids

            # try:
            torch.cuda.synchronize()
            start_time = time.time()

            output_ids, new_token, idx = draft_forward(
                torch.as_tensor(input_ids).cuda(),
                model,
                tokenizer,
                tree_choices,
            )
            torch.cuda.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            conv.stop_str = "</s>"
            if conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()

            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output
    print('Warmup done')

    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template("vicuna")
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer([prompt]).input_ids

                try:
                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids, new_token, idx = draft_forward(
                        torch.as_tensor(input_ids).cuda(),
                        model,
                        tokenizer,
                        tree_choices,
                    )
                    torch.cuda.synchronize()
                    total_time = time.time() - start_time
                    output_ids = output_ids[0][len(input_ids[0]):]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--draft-model-path",
        type=str,
        default="down_checkpoints/LC70B",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama2chat/70B/",
                        help="1")
    parser.add_argument(
        "--load-in-8bit", action="store_false", help="Use 8-bit quantization"
    )
    parser.add_argument("--model-id", type=str, default="ess-vicuna-70b-fp16")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument(
        "--use-safetensor-weight",
        type=bool,
        default=False,
    )
    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    args.tree_choices = eval(args.tree_choices)
    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    use_safetensor_weight = args.use_safetensor_weight
    run_eval(
        args.base_model_path,
        args.draft_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,

        args.temperature,
        args.tree_choices,
    )

    reorg_answer_file(answer_file)
