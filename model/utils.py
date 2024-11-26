import copy
import random

from typing import List, Tuple
import time
import torch


TOPK = 10  # topk for sparse tree

def timer(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f'{func.__name__} took {elapsed} seconds')
        return result

    return wrapper


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_medusa_buffers(medusa_choices, device="cuda"):
    """
    Generate buffers for the Medusa structure based on the provided choices.
    
    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".
    
    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
        }
    
    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v,  device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers

def initialize_tree(input_ids, model, tree_attn_mask, past_key_values):
    tree_logits, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True
    )
    model.base_model.model.tree_mask = tree_attn_mask
    return tree_logits, logits, hidden_state, sample_token


def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits[0]

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]

    cart_candidates_prob = None
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, cart_candidates_prob, tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
        new_attn_mask
):
    position_ids = tree_position_ids + input_ids.shape[1]
    model.base_model.model.tree_mask = new_attn_mask
    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
        init=False,
    )
    #only base model
    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs


def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        cart_candidates_prob,
        op,
        p_indices,
        tree_candidates,
        b_indices
) -> Tuple[torch.Tensor, int]:
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    # Find the tokens that match the maximum logits for each position in the sequence
    posterior_mask = (
            candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
    ).int()
    candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
    accept_length = candidates_accept_length.max()
    # Choose the best candidate
    if accept_length == 0:
        # Default to the first candidate if none are accepted
        best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
    else:
        best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
    return best_candidate, accept_length, logits[best_candidate, accept_length]
    


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits,
        tree_logits,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state,
        hidden_state_new,
        sample_p
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    
    prob = sample_p
    token = torch.argmax(prob)
    token = token[None, None]

    tree_logits = model.layer.topK_genrate(accept_hidden_state_new,
                                              input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
                                              head=model.base_model.lm_head)

    new_token += accept_length + 1

    return input_ids, tree_logits, new_token, None, token

