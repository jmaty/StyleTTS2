import os
import os.path as osp
from collections import OrderedDict

import torch
import yaml
from transformers import AlbertConfig, AlbertModel

from logger import get_logger

# Setup logger
logger = get_logger(__name__)


class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path, encoding="utf-8"))

    albert_base_configuration = AlbertConfig(**plbert_config["model_params"])
    bert = CustomAlbert(albert_base_configuration)

    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"):
            ckpts.append(f)

    iters = [
        int(f.split("_")[-1].split(".")[0])
        for f in ckpts
        if os.path.isfile(os.path.join(log_dir, f))
    ]
    iters = sorted(iters)[-1]

    checkpoint_path = osp.join(log_dir, f"step_{iters}.t7")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    state_dict = checkpoint["net"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        if name.startswith("encoder."):
            name = name[8:]  # remove `encoder.`
            new_state_dict[name] = v
    try:
        del new_state_dict["embeddings.position_ids"]
    except KeyError:
        pass
    bert.load_state_dict(new_state_dict, strict=False)

    logger.info(
        "Loading PL-BERT with size %d from %s", bert.config.max_position_embeddings, checkpoint_path
    )

    return bert
