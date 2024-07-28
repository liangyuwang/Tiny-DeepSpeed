# Copyright (c) 2024 liangyuwang
# Licensed under the Apache License, Version 2.0

import torch
from collections import OrderedDict

def partition_tensors(
        tensors_dict: OrderedDict, 
        ranks_map: list = None,
        num_parts: int = None, 
        evenness_priority: float = 0, 
        malloc=False,
        verbose=False,
    ):
    """
    Partition the tensors of a model into multiple parts for distributed training.
    This function uses an 'evenness_priority' to control the balance between evenly distributing
    tensors across partitions and keeping closely related tensors (e.g., those from the same layer or neighbor layers)
    together within the same partition.

    Args:
        tensors_dict: OrderedDict, the dict of model tensors (could be on meta device) to partition.
        ranks_map: list, optional, a list of rank identifiers, each corresponding to a partition.
        num_parts: int, the number of parts to partition the model into.
        evenness_priority: float, a priority value ranging from 0 to 1 that determines the balance 
                        between evenness of tensor distribution and keeping related tensors together.
                        - A value of 0 prioritizes keeping related tensors together as much as possible,
                            potentially leading to some partitions being significantly larger or smaller than others.
                        - A value of 1 prioritizes even distribution of tensors, ensuring that each partition
                            is as close as possible to having the same number of tensors, even if it means splitting
                            closely related tensors across different partitions.
                        - Values in between adjust the sensitivity to these factors dynamically.
                        - Tips: 
    Returns:
        parts: dict, a dictionary containing the partition, with each partition holding a list of tensor names
            and their 

    Note:
        This method ensures that all partitions are used, but an unevenness in tensor sizes may lead to non-ideal
        distributions. Warnings are printed if any partitions are empty, indicating unused computational resources.

    Tip:
        Setting the 'evenness_priority':
        - For a small number of parts (e.g., 2-4), setting 'evenness_priority' closer to 0 helps minimize the risk of
        underutilizing any computational resource by keeping more related tensors together.
        - As the number of parts increases, consider increasing the 'evenness_priority' towards 1 to ensure a more
        uniform distribution of tensors across all computational resources, which can be crucial for efficiency in
        large-scale distributed training environments.
    """
    assert 0 <= evenness_priority <= 1, "Evenness priority must be between 0 and 1"
    if ranks_map:
        num_parts = len(ranks_map)
    else:
        assert num_parts > 0, "Number of parts must be a positive integer"
    if malloc:
        assert ranks_map, "Ranks map must be provided if malloc is set to True"
        if verbose: print("The tensors will be moved to the corresponding ranks devices based on the ranks map.")

    # Collect all tensors
    tensors = list(tensors_dict.items())

    # Calculate total number of tensors
    total_tensors = sum(p.numel() for _, p in tensors)

    # Target number of tensors per part
    target_per_part = total_tensors / num_parts

    # Initialize parts and their current sizes
    parts = {i: [] for i in range(num_parts)}
    parts_sizes = {i: 0 for i in range(num_parts)}
    part_assignment = {}

    current_part = 0
    for name, tensor in tensors:
        # Calculate current threshold based on evenness_priority
        current_threshold = target_per_part * (1 + evenness_priority * (parts_sizes[current_part] / target_per_part - 1))

        # Check if adding this tensor to the current part exceeds the dynamically adjusted threshold
        if parts_sizes[current_part] + tensor.numel() > current_threshold and parts_sizes[current_part] != 0:
            current_part = min(current_part + 1, num_parts - 1)
        
        # Add tensor to the current part and update the assignment map
        parts[current_part].append((name, tensor.numel()))
        parts_sizes[current_part] += tensor.numel()
        part_assignment[name] = current_part

        # Move tensor to its corresponding rank if a rank map is provided
        if malloc:
            if tensor.device.type != "meta":
                tensor = tensor.to(ranks_map[current_part])
            else:
                tensor = torch.empty(tensor.size(), device=ranks_map[current_part], dtype=tensor.dtype)
        tensors_dict[name] = tensor
        if verbose: print(f"partition {name} to \t rank {ranks_map[current_part]}")

    # Check for any empty parts and issue warnings
    for part, items in parts.items():
        if not items:
            warning = f"Warning: Part {part} is empty. Consider adjusting the evenness_priority or the number of parts."
            if verbose: print(warning)

    return part_assignment, tensors_dict





if __name__=="__main__":
    from torchvision.models import vit_b_16
    
    with torch.device('meta'):
        model = vit_b_16()
    ranks_map = ["cuda:0", "cuda:1"]
    parts, _ = partition_tensors(OrderedDict(model.named_parameters()), 
                                 ranks_map=ranks_map,
                                 evenness_priority=0,
                                 verbose=True)

    print("-------------------")

    with torch.device('meta'):
        model = vit_b_16()
    parts, _ = partition_tensors(OrderedDict(model.named_parameters()), 
                                 num_parts=8,
                                 evenness_priority=0,
                                 verbose=True)