import numpy as np
import torch
import torch.utils.data

from pbdl.torch.dataset import Dataset

def _collate_fn_(batch):
    """
    Concatenates data arrays with inflated constant layers and stacks them into batches.

    Returns:
        torch.Tensor: Data batch tensor
        torch.Tensor: Target batch tensor
    """

    data = np.stack(  # stack batch items
        [
            np.concatenate(  # concatenate data and constant layers
                [item[0]]
                + [
                    [
                        np.full_like(
                            item[0][0], constant
                        )  # inflate constants to constant layers
                    ]
                    for constant in item[2]
                ],
                axis=0,
            )
            for item in batch
        ],
        axis=0,
    )

    targets = np.stack([item[1] for item in batch])

    return torch.tensor(data), torch.tensor(targets)

class Dataloader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):

        # dispatch arguments
        loader_kwargs = {
            k: kwargs.pop(k)
            for k in [
                "batch_size",
                "shuffle",
                "sampler",
                "batch_sampler",
                "num_workers",
                "collate_fn",
                "pin_memory",
                "drop_last",
                "timeout",
                "worker_init_fn",
                "multiprocessing_context",
                "generator",
                "prefetch_factor",
                "persistent_workers",
                "pin_memory_device",
            ]
            if k in kwargs
        }
        dset_kwargs = {
            k: kwargs.pop(k)
            for k in [
                "intermediate_time_steps",
                "normalize",
                "sel_sims",
                "trim_start",
                "trim_end",
                "step_size",
                "disable_progress",
            ]
            if k in kwargs
        }
        
        dataset = Dataset(
            *args[:2], **dset_kwargs, **kwargs
        )  # remaining kwargs are expected to be config parameters

        if "collate_fn" not in loader_kwargs:
            loader_kwargs["collate_fn"] = _collate_fn_

        super().__init__(dataset, **loader_kwargs)
