import logging
import torch
from omegaconf import OmegaConf

from minigpt4.common.registry import registry
from minigpt4.models.base_model import BaseModel
from minigpt4.models.blip2 import Blip2Base


__all__ = [
    "load_model",
    "BaseModel",
    "Blip2Base",
    "MiniGPT4",
]


def load_model(name, model_type, is_eval=False, device="cpu", checkpoint=None):
    

    model = registry.get_model_class(name).from_pretrained(model_type=model_type)

    if checkpoint is not None:
        model.load_checkpoint(checkpoint)

    if is_eval:
        model.eval()

    if device == "cpu":
        model = model.float()

    return model.to(device)

class ModelZoo:
    

    def __init__(self) -> None:
        self.model_zoo = {
            k: list(v.PRETRAINED_MODEL_CONFIG_DICT.keys())
            for k, v in registry.mapping["model_name_mapping"].items()
        }

    def __str__(self) -> str:
        return (
            "=" * 50
            + "\n"
            + f"{'Architectures':<30} {'Types'}\n"
            + "=" * 50
            + "\n"
            + "\n".join(
                [
                    f"{name:<30} {', '.join(types)}"
                    for name, types in self.model_zoo.items()
                ]
            )
        )

    def __iter__(self):
        return iter(self.model_zoo.items())

    def __len__(self):
        return sum([len(v) for v in self.model_zoo.values()])


model_zoo = ModelZoo()