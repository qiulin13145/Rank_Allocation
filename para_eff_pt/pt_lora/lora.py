import os
import math
import json

from typing import List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoConfig


@dataclass
class LoRaConfig:
    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: List[str]


class LoRaModel(torch.nn.Module):
    def __init__(
        self,
        model,
        *,
        target_modules,
        r=128,
        lora_alpha=32,
        lora_dropout=0.1,
        trainable_scaling=False,
    ):
        if r < 0:
            raise ValueError("r must be nonnegative.")

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.trainable_scaling = trainable_scaling
        self.parameterized_modules = []

        self._config = LoRaConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )

        # patch methods
        self.forward = self.wrapped_model.forward

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            weight_data = module.weight.data
            bias_data = None
            if module.bias is not None:
                bias_data = module.bias.data

            print(f"Reparameterized module: {module_name}")
            # self.parameterized_modules.append(module_name)
            new_module = LoRaLinear(
                module.in_features,
                module.out_features,
                r=self.r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                trainable_scaling=self.trainable_scaling,
                weight_data=weight_data,
                bias_data=bias_data,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )

            # module.weight = None
            del module

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        parent = self.wrapped_model.get_submodule(parent_name)
        return parent

    def save_pretrained(self, path, max_shard_size="100GB"):
        # TODO
        self.wrapped_model.save_pretrained(path)
        with open(os.path.join(path, "lora_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        # TODO
        with open(os.path.join(path, "lora_config.json"), "r") as f:
            splora_config = json.load(f)

        config = AutoConfig.from_pretrained(path)

        base_model = AutoModelForCausalLM.from_config(config)
        if "keep_original" in splora_config:
            print("WARNING: keep_original is deprecated. Use lora_only instead.")
            print(f"keep_original: {splora_config['keep_original']}")
            splora_config["lora_only"] = not splora_config.pop("keep_original")
            splora_config["keep_original_weights"] = not splora_config["lora_only"]

        if "trainable_scaling" not in splora_config:
            splora_config["trainable_scaling"] = False

        model = cls(base_model, **splora_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model


class LoRaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        trainable_scaling: bool = False,
        weight_data=None,
        bias_data=None,
        bias=True,
        device=None,
        dtype=None,
    ):
        """
        Reparameterized sparse and low rank linear layer
                    x W_a @ W_b * lora_alpha / r + x W_sp + bias
        Notice that scale = lora_alpha / r.
        Notice that this class cannot be wrapped to linear layer and thus cannot be used for fine-tune
        For fine-tune, please refer to ... TODO
        """
        super().__init__()
        # nn.Module.__init__(self)
        if r <= 0:
            raise ValueError("r must be positive.")

        if bias_data is None:
            bias_data = (
                torch.zeros(
                    out_features, device=device, dtype=dtype, requires_grad=True
                )
                if bias
                else None
            )
        self.bias = nn.Parameter(bias_data) if bias else None

        if weight_data is None:
            # note that our trainable weight are W_a and W_b
            weight_data = torch.zeros(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=False,
            )
        self.weight = nn.Parameter(weight_data, requires_grad=False)

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.trainable_scaling = trainable_scaling
        self.device = device
        self.dtype = dtype

        self.lora_A = nn.Linear(
            in_features=in_features,
            out_features=r,
            bias=False,
            dtype=dtype,
            device=device,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        self.lora_B = nn.Linear(
            in_features=r,
            out_features=out_features,
            bias=False,
            dtype=dtype,
            device=device,
        )
        # nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))
        
        
        if trainable_scaling:
            self.scaling = nn.Parameter(
                torch.tensor([1.0], device=device, dtype=dtype), requires_grad=True
            )
        else:
            self.scaling = self.lora_alpha / self.r

    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()

        return self.scaling

    def forward(self, x):
        out = F.linear(x, self.weight, bias=self.bias)

        if self.r > 0:
            out += (
                self.lora_B(self.lora_A(self.lora_dropout(x))) * self._post_lora_scale()
            )

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, rank={self.r}, "
            f"bias={self.bias is not None}"
        )
