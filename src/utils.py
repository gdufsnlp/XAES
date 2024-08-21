import os
import random
import subprocess

import numpy as np
import torch


def fix_seed(seed: int):
    """Fix the random seed."""

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_git_revision_hash() -> str:
    """Obtain the current git commit id."""

    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        return f"{branch}:{commit_id}"
    except:
        return ""


def set_device(data_dict: dict, device: torch.device):
    for k, v in data_dict.items():
        if type(v) == torch.Tensor:
            data_dict[k] = v.to(device)
        else:
            data_dict[k] = set_device(
                data_dict=v,
                device=device
            )
    return data_dict
