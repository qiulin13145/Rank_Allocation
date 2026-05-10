import json
import math
import os
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


@dataclass
class RankAllocationLoRaConfig:
    r: int
    lora_alpha: float
    lora_dropout: float
    target_modules: List[str]
    trainable_scaling: bool = False


@dataclass
class RankMove:
    remove_module: "RankAllocationLoRaLinear"
    add_module: "RankAllocationLoRaLinear"
    delta_rank: int


@dataclass
class RankAllocationResult:
    old_ranks: Dict["RankAllocationLoRaLinear", int]
    new_ranks: Dict["RankAllocationLoRaLinear", int]
    moves: List[RankMove]

    @property
    def changed_modules(self):
        return [
            module
            for module, old_rank in self.old_ranks.items()
            if self.new_ranks[module] != old_rank
        ]

    @property
    def rank_moved(self):
        return sum(move.delta_rank for move in self.moves)


class RankAllocationLoRaModel(torch.nn.Module):
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
        if r <= 0:
            raise ValueError("r must be positive.")

        super().__init__()
        self.wrapped_model: nn.Module = model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules
        self.trainable_scaling = trainable_scaling
        self.parameterized_modules = []

        target_modules_list = target_modules
        if isinstance(target_modules_list, str):
            target_modules_list = [target_modules_list]

        self._config = RankAllocationLoRaConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(target_modules_list),
            trainable_scaling=trainable_scaling,
        )

        self.forward = self.wrapped_model.forward

        for module_name, module in self.wrapped_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(target_key in module_name for target_key in target_modules_list):
                continue

            bias_data = module.bias.data if module.bias is not None else None
            print(f"Reparameterized module: {module_name}")
            new_module = RankAllocationLoRaLinear(
                module.in_features,
                module.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                trainable_scaling=trainable_scaling,
                bias=module.bias is not None,
                bias_data=bias_data,
                device=module.weight.device,
                dtype=module.weight.dtype,
                module_name=module_name,
            )

            parent = self._get_parent(module_name)
            module_suffix = module_name.split(".")[-1]
            setattr(parent, module_suffix, new_module)

        torch.cuda.empty_cache()

    def _get_parent(self, module_name):
        module_names_list = module_name.split(".")
        parent_name = ".".join(module_names_list[:-1])
        return self.wrapped_model.get_submodule(parent_name)

    def save_pretrained(self, path, max_shard_size="100GB"):
        self.wrapped_model.save_pretrained(path, safe_serialization=False)
        with open(os.path.join(path, "rank_allocation_lora_config.json"), "w") as f:
            json.dump(self._config.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "rank_allocation_lora_config.json"), "r") as f:
            lora_config = json.load(f)

        config = AutoConfig.from_pretrained(path)
        base_model = AutoModelForCausalLM.from_config(config)
        model = cls(base_model, **lora_config)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            state_dict = torch.load(f, map_location="cpu")

        model.wrapped_model.load_state_dict(state_dict, strict=True)
        return model


class RankAllocationLoRaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        *,
        lora_alpha: float = 1,
        lora_dropout: float = 0.0,
        trainable_scaling: bool = False,
        bias=True,
        bias_data=None,
        device=None,
        dtype=None,
        module_name: Optional[str] = None,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = r
        self.r = r
        self.initial_rank = r
        self.lora_alpha = lora_alpha
        self.base_scaling = self.lora_alpha / self.initial_rank
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.trainable_scaling = trainable_scaling
        self.device = device
        self.dtype = dtype
        self.module_name = module_name or ""

        if bias:
            if bias_data is None:
                bias_data = torch.zeros(out_features, device=device, dtype=dtype)
            self.bias = nn.Parameter(bias_data)
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(
            torch.empty(r, in_features, dtype=dtype, device=device),
            requires_grad=True,
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(
            torch.empty(out_features, r, dtype=dtype, device=device),
            requires_grad=True,
        )
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

        if trainable_scaling:
            self.scaling = nn.Parameter(
                torch.tensor([1.0], device=device, dtype=dtype),
                requires_grad=True,
            )
        else:
            self.scaling = self.base_scaling

        self.credit_ema = 0.0
        self.energy_ema = 0.0
        self.eff_ema = 0.0
        self.probe_ema = 0.0
        self.score = 0.0
        self.last_credit = 0.0
        self.last_energy = 0.0
        self.last_probe = 0.0
        self._credit_snapshot = None
        self.probe_enabled = False

    def _post_lora_scale(self):
        if self.trainable_scaling:
            return self.scaling.tanh()
        return self.scaling

    def _scale_value(self):
        scale = self._post_lora_scale()
        if torch.is_tensor(scale):
            return scale.detach().to(device=self.lora_A.device, dtype=self.lora_A.dtype)
        return torch.tensor(scale, device=self.lora_A.device, dtype=self.lora_A.dtype)

    def forward(self, x):
        hidden = F.linear(self.lora_dropout(x), self.lora_A * self._post_lora_scale())
        out = F.linear(hidden, self.lora_B, self.bias)

        if self.probe_enabled:
            probe_hidden = F.linear(x, self.A_probe)
            out = out + F.linear(probe_hidden, self.B_probe)

        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}"
        )

    def enable_probe(self, probe_rank=1, sigma=1e-3):
        if self.probe_enabled:
            self.disable_probe()
        self.B_probe = nn.Parameter(
            torch.zeros(
                self.out_features,
                probe_rank,
                device=self.lora_B.device,
                dtype=self.lora_B.dtype,
            )
        )
        self.A_probe = nn.Parameter(
            torch.randn(
                probe_rank,
                self.in_features,
                device=self.lora_A.device,
                dtype=self.lora_A.dtype,
            )
            * sigma
        )
        self.probe_enabled = True

    def get_probe_score(self):
        if not self.probe_enabled or self.B_probe.grad is None:
            return 0.0
        score = self.B_probe.grad.detach().float().norm().item()
        denom = self.A_probe.detach().float().norm().item() + 1e-12
        return score / denom

    def disable_probe(self):
        if hasattr(self, "B_probe"):
            del self.B_probe
        if hasattr(self, "A_probe"):
            del self.A_probe
        self.probe_enabled = False

    def snapshot_credit_state(self):
        if self.lora_A.grad is None or self.lora_B.grad is None:
            self._credit_snapshot = None
            return False
        self._credit_snapshot = {
            "A": self.lora_A.detach().clone(),
            "B": self.lora_B.detach().clone(),
            "A_grad": self.lora_A.grad.detach().clone(),
            "B_grad": self.lora_B.grad.detach().clone(),
        }
        return True

    def update_credit_from_snapshot(self, beta=0.95, eps=1e-12):
        if self._credit_snapshot is None:
            return None

        dA = self.lora_A.detach() - self._credit_snapshot["A"]
        dB = self.lora_B.detach() - self._credit_snapshot["B"]
        credit = (self._credit_snapshot["A_grad"].float() * dA.float()).sum()
        credit = credit + (self._credit_snapshot["B_grad"].float() * dB.float()).sum()
        energy = dA.float().pow(2).sum() + dB.float().pow(2).sum()

        credit_value = credit.item()
        energy_value = energy.item()
        positive_credit = max(-credit_value, 0.0)
        efficiency = positive_credit / (energy_value + eps)

        self.credit_ema = beta * self.credit_ema + (1.0 - beta) * positive_credit
        self.energy_ema = beta * self.energy_ema + (1.0 - beta) * energy_value
        self.eff_ema = beta * self.eff_ema + (1.0 - beta) * efficiency
        self.last_credit = credit_value
        self.last_energy = energy_value
        self._credit_snapshot = None
        return credit_value, energy_value, efficiency

    def compute_rank_score(self, eps=1e-12):
        scarcity = 1.0 / math.sqrt(float(self.rank) + eps)
        self.score = self.eff_ema * self.probe_ema * scarcity
        return self.score

    @torch.no_grad()
    def refactorize_with_rank(self, new_rank):
        if new_rank <= 0:
            raise ValueError("new_rank must be positive.")

        old_dtype = self.lora_A.dtype
        old_device = self.lora_A.device
        scale = self._scale_value().float()
        scale_value = float(scale.item()) if scale.numel() == 1 else float(scale.mean().item())
        if abs(scale_value) < 1e-12:
            scale_value = 1.0

        W = self.lora_B.detach().float().mm(
            (self.lora_A.detach().float() * scale_value)
        )
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        rank = min(new_rank, U.shape[1], Vh.shape[0])
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        sqrt_S = torch.sqrt(S_r.clamp_min(0.0))

        new_B = U_r * sqrt_S.unsqueeze(0)
        new_A_effective = sqrt_S.unsqueeze(1) * Vh_r
        new_scale = self.base_scaling
        if self.trainable_scaling:
            new_scale = scale_value
        new_A = new_A_effective / new_scale

        self.rank = rank
        self.r = rank
        self.lora_A = nn.Parameter(new_A.to(device=old_device, dtype=old_dtype))
        self.lora_B = nn.Parameter(new_B.to(device=old_device, dtype=old_dtype))
        if not self.trainable_scaling:
            self.scaling = self.base_scaling
        self._credit_snapshot = None


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def list_rank_allocation_lora_modules(model) -> List[RankAllocationLoRaLinear]:
    unwrapped = unwrap_model(model)
    modules = []
    for name, module in unwrapped.named_modules():
        if isinstance(module, RankAllocationLoRaLinear):
            if not module.module_name:
                module.module_name = name
            modules.append(module)
    return modules


def snapshot_credit_state(modules: Iterable[RankAllocationLoRaLinear]):
    return sum(1 for module in modules if module.snapshot_credit_state())


def update_credit_stats(
    modules: Iterable[RankAllocationLoRaLinear],
    beta=0.95,
):
    updated = 0
    for module in modules:
        if module.update_credit_from_snapshot(beta=beta) is not None:
            updated += 1
    return updated


def enable_rank_probes(modules: Iterable[RankAllocationLoRaLinear], probe_rank=1, sigma=1e-3):
    for module in modules:
        module.enable_probe(probe_rank=probe_rank, sigma=sigma)


def update_probe_stats(
    modules: Iterable[RankAllocationLoRaLinear],
    beta=0.9,
    distributed=True,
):
    updated_scores = []
    for module in modules:
        score = module.get_probe_score()
        if (
            distributed
            and dist.is_available()
            and dist.is_initialized()
            and module.lora_A.is_cuda
        ):
            score_tensor = torch.tensor(score, device=module.lora_A.device)
            dist.all_reduce(score_tensor, op=dist.ReduceOp.SUM)
            score = (score_tensor / dist.get_world_size()).item()
        module.last_probe = score
        module.probe_ema = beta * module.probe_ema + (1.0 - beta) * score
        updated_scores.append(score)
    return updated_scores


def disable_rank_probes(modules: Iterable[RankAllocationLoRaLinear]):
    for module in modules:
        module.disable_probe()


def compute_rank_scores(modules: Iterable[RankAllocationLoRaLinear]):
    return [module.compute_rank_score() for module in modules]


def _rank_limit(module, ratio):
    base_dim = min(module.in_features, module.out_features)
    return max(1, int(math.floor(base_dim * ratio)))


def allocate_rank_budget(
    modules: List[RankAllocationLoRaLinear],
    delta_rank=8,
    top_k=4,
    min_ratio=0.125,
    max_ratio=0.5,
    hysteresis=0.1,
):
    old_ranks = {module: module.rank for module in modules}
    new_ranks = dict(old_ranks)
    moves = []

    if not modules or delta_rank <= 0 or top_k <= 0:
        return RankAllocationResult(old_ranks=old_ranks, new_ranks=new_ranks, moves=moves)

    add_list = sorted(modules, key=lambda module: module.score, reverse=True)
    remove_list = sorted(modules, key=lambda module: module.score)
    add_idx = 0
    remove_idx = 0

    while len(moves) < top_k and add_idx < len(add_list) and remove_idx < len(remove_list):
        add_module = add_list[add_idx]
        remove_module = remove_list[remove_idx]

        if add_module is remove_module:
            add_idx += 1
            continue

        add_max = _rank_limit(add_module, max_ratio)
        remove_min = _rank_limit(remove_module, min_ratio)

        if new_ranks[add_module] + delta_rank > add_max:
            add_idx += 1
            continue
        if new_ranks[remove_module] - delta_rank < remove_min:
            remove_idx += 1
            continue
        if add_module.score <= remove_module.score * (1.0 + hysteresis):
            break

        new_ranks[add_module] += delta_rank
        new_ranks[remove_module] -= delta_rank
        moves.append(RankMove(remove_module, add_module, delta_rank))
        add_idx += 1
        remove_idx += 1

    return RankAllocationResult(old_ranks=old_ranks, new_ranks=new_ranks, moves=moves)


@torch.no_grad()
def apply_rank_allocation(result: RankAllocationResult):
    for module, new_rank in result.new_ranks.items():
        module.refactorize_with_rank(new_rank)


def _median(values):
    return statistics.median(values) if values else 0.0


def _module_kind(module: RankAllocationLoRaLinear):
    name = module.module_name.lower()
    if "attn" in name or "attention" in name:
        return "attention"
    if "mlp" in name:
        return "mlp"
    return "other"


def summarize_rank_allocation(
    modules: List[RankAllocationLoRaLinear],
    result: RankAllocationResult,
    step: int,
    loss_value: float,
    lr: float,
    top_n=5,
):
    ranks = [module.rank for module in modules]
    scores = [module.score for module in modules]
    probes = [module.probe_ema for module in modules]
    credit_effs = [module.eff_ema for module in modules]
    energies = [module.energy_ema for module in modules]

    metrics = {
        "rank_allocation/rank_moved": result.rank_moved,
        "rank_allocation/changed_modules": len(result.changed_modules),
        "rank_allocation/rank_mean": sum(ranks) / max(len(ranks), 1),
        "rank_allocation/rank_min": min(ranks) if ranks else 0,
        "rank_allocation/rank_max": max(ranks) if ranks else 0,
        "rank_allocation/score_median": _median(scores),
        "rank_allocation/score_max": max(scores) if scores else 0.0,
        "rank_allocation/probe_median": _median(probes),
        "rank_allocation/probe_max": max(probes) if probes else 0.0,
        "rank_allocation/credit_eff_median": _median(credit_effs),
        "rank_allocation/credit_eff_max": max(credit_effs) if credit_effs else 0.0,
        "rank_allocation/energy_median": _median(energies),
        "rank_allocation/energy_max": max(energies) if energies else 0.0,
    }

    by_kind = {}
    for module in modules:
        by_kind.setdefault(_module_kind(module), []).append(module.rank)
    for kind, kind_ranks in by_kind.items():
        metrics[f"rank_allocation/{kind}_rank_mean"] = sum(kind_ranks) / len(kind_ranks)

    ordered = sorted(modules, key=lambda module: module.score, reverse=True)
    top_modules = ordered[:top_n]
    bottom_modules = list(reversed(ordered[-top_n:])) if ordered else []

    table_rows = []
    for label, selected in (("top", top_modules), ("bottom", bottom_modules)):
        for module in selected:
            table_rows.append(
                [
                    label,
                    module.module_name,
                    result.old_ranks.get(module, module.rank),
                    result.new_ranks.get(module, module.rank),
                    module.score,
                    module.eff_ema,
                    module.probe_ema,
                    module.energy_ema,
                ]
            )

    changed_rows = [
        [
            move.remove_module.module_name,
            result.old_ranks[move.remove_module],
            result.new_ranks[move.remove_module],
            move.add_module.module_name,
            result.old_ranks[move.add_module],
            result.new_ranks[move.add_module],
            move.delta_rank,
        ]
        for move in result.moves
    ]

    lines = [
        (
            f"[rank_allocation_lora] restart step={step} loss={loss_value:.6f} "
            f"lr={lr:.6g} total_rank={sum(ranks)} changed={len(result.changed_modules)} "
            f"rank_moved={result.rank_moved} rank(min/mean/max)="
            f"{metrics['rank_allocation/rank_min']}/"
            f"{metrics['rank_allocation/rank_mean']:.2f}/"
            f"{metrics['rank_allocation/rank_max']}"
        ),
        (
            f"[rank_allocation_lora] score median/max="
            f"{metrics['rank_allocation/score_median']:.4e}/"
            f"{metrics['rank_allocation/score_max']:.4e} "
            f"credit_eff median/max="
            f"{metrics['rank_allocation/credit_eff_median']:.4e}/"
            f"{metrics['rank_allocation/credit_eff_max']:.4e} "
            f"probe median/max="
            f"{metrics['rank_allocation/probe_median']:.4e}/"
            f"{metrics['rank_allocation/probe_max']:.4e} "
            f"energy median/max="
            f"{metrics['rank_allocation/energy_median']:.4e}/"
            f"{metrics['rank_allocation/energy_max']:.4e}"
        ),
    ]

    if by_kind:
        kind_summary = ", ".join(
            f"{kind}={sum(kind_ranks) / len(kind_ranks):.2f}"
            for kind, kind_ranks in sorted(by_kind.items())
        )
        lines.append(f"[rank_allocation_lora] mean rank by module type: {kind_summary}")

    for row in changed_rows[:top_n]:
        lines.append(
            "[rank_allocation_lora] move "
            f"-{row[6]} {row[0]} {row[1]}->{row[2]} | "
            f"+{row[6]} {row[3]} {row[4]}->{row[5]}"
        )

    return {
        "metrics": metrics,
        "table_rows": table_rows,
        "changed_rows": changed_rows,
        "lines": lines,
    }
