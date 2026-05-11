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
    rank_growth_init_std: float = 1e-4


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
    rejection_stats: Dict[str, int]

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
        rank_growth_init_std=1e-4,
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
        self.rank_growth_init_std = rank_growth_init_std
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
            rank_growth_init_std=rank_growth_init_std,
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
                rank_growth_init_std=rank_growth_init_std,
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
        rank_growth_init_std: float = 1e-4,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive.")
        if rank_growth_init_std <= 0:
            raise ValueError("rank_growth_init_std must be positive.")

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
        self.rank_growth_init_std = rank_growth_init_std

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
        self.last_A_probe = None
        self.add_score = 0.0
        self.remove_score = 0.0

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
        W = torch.matmul(self.lora_B, self.lora_A * self._post_lora_scale())
        out = F.linear(x, W, None if self.bias is None else self.bias)

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
        probe = torch.randn(
            probe_rank,
            self.in_features,
            device=self.lora_A.device,
            dtype=torch.float32,
        )
        Q, _ = torch.linalg.qr(self.lora_A.detach().float().T, mode="reduced")
        probe = probe - (probe @ Q) @ Q.T
        probe = probe / (probe.norm(dim=1, keepdim=True) + 1e-12)
        probe = probe * sigma
        self.A_probe = nn.Parameter(probe.to(dtype=self.lora_A.dtype))
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


    def capture_probe_basis(self):
        if not self.probe_enabled or not hasattr(self, "A_probe"):
            return
        self.last_A_probe = self.A_probe.detach().clone()

    @torch.no_grad()
    def rank_tail_energy(self, new_rank: int) -> float:
        W = self.lora_B.detach().float().mm(self.lora_A.detach().float())
        S = torch.linalg.svdvals(W)
        if S.numel() == 0 or new_rank >= S.numel():
            return 0.0
        total = S.pow(2).sum().sqrt()
        tail = S[new_rank:].pow(2).sum().sqrt()
        return (tail / (total + 1e-12)).item()

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
        self.add_score = self.probe_ema + 0.5 * self.eff_ema + 0.1 * scarcity
        self.remove_score = self.eff_ema + self.probe_ema
        self.score = self.add_score
        return self.score

    @torch.no_grad()
    def refactorize_with_rank(self, new_rank):
        if new_rank <= 0:
            raise ValueError("new_rank must be positive.")

        old_rank = self.rank
        old_dtype = self.lora_A.dtype
        old_device = self.lora_A.device
        W = self.lora_B.detach().float().mm(self.lora_A.detach().float())
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        target_rank = min(new_rank, U.shape[1], Vh.shape[0])
        svd_rank = min(target_rank, old_rank) if new_rank > old_rank else target_rank
        U_r = U[:, :svd_rank]
        S_r = S[:svd_rank]
        Vh_r = Vh[:svd_rank, :]
        sqrt_S = torch.sqrt(S_r.clamp_min(0.0))

        new_B = U_r * sqrt_S.unsqueeze(0)
        new_A = sqrt_S.unsqueeze(1) * Vh_r

        if target_rank > svd_rank:
            extra_rank = target_rank - svd_rank
            extra_B = torch.zeros(self.out_features, extra_rank, device=old_device, dtype=old_dtype)
            if self.last_A_probe is not None and self.last_A_probe.shape[1] == self.in_features:
                probe_rows = min(extra_rank, self.last_A_probe.shape[0])
                extra_A = self.last_A_probe[:probe_rows].to(device=old_device, dtype=old_dtype).clone()
                if probe_rows < extra_rank:
                    rem = torch.empty(extra_rank - probe_rows, self.in_features, device=old_device, dtype=old_dtype)
                    rem.normal_(mean=0.0, std=self.rank_growth_init_std)
                    extra_A = torch.cat([extra_A, rem], dim=0)
            else:
                extra_A = torch.empty(extra_rank, self.in_features, device=old_device, dtype=old_dtype)
                extra_A.normal_(mean=0.0, std=self.rank_growth_init_std)
            new_A = torch.cat([new_A.to(device=old_device, dtype=old_dtype), extra_A], dim=0)
            new_B = torch.cat([new_B.to(device=old_device, dtype=old_dtype), extra_B], dim=1)
        else:
            new_A = new_A.to(device=old_device, dtype=old_dtype)
            new_B = new_B.to(device=old_device, dtype=old_dtype)

        self.rank = target_rank
        self.r = target_rank
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
        module.capture_probe_basis()
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
    tail_threshold=5e-3,
):
    old_ranks = {module: module.rank for module in modules}
    new_ranks = dict(old_ranks)
    moves = []
    rejection_stats = {
        "add_max": 0,
        "remove_min": 0,
        "same_module": 0,
        "remove_already_used": 0,
        "hysteresis": 0,
        "tail_energy": 0,
    }

    if not modules or delta_rank <= 0 or top_k <= 0:
        return RankAllocationResult(old_ranks=old_ranks, new_ranks=new_ranks, moves=moves, rejection_stats=rejection_stats)

    add_list = sorted(modules, key=lambda module: module.add_score, reverse=True)
    remove_list = sorted(modules, key=lambda module: module.remove_score)

    for add_module in add_list:
        if len(moves) >= top_k:
            break
        add_max = _rank_limit(add_module, max_ratio)
        if new_ranks[add_module] + delta_rank > add_max:
            rejection_stats["add_max"] += 1
            continue

        for remove_module in remove_list:
            if add_module is remove_module:
                rejection_stats["same_module"] += 1
                continue
            if any(move.remove_module is remove_module for move in moves):
                rejection_stats["remove_already_used"] += 1
                continue

            remove_min = _rank_limit(remove_module, min_ratio)
            if new_ranks[remove_module] - delta_rank < remove_min:
                rejection_stats["remove_min"] += 1
                continue
            if (add_module.add_score - remove_module.remove_score) <= hysteresis:
                rejection_stats["hysteresis"] += 1
                continue
            if remove_module.rank_tail_energy(new_ranks[remove_module] - delta_rank) > tail_threshold:
                rejection_stats["tail_energy"] += 1
                continue

            new_ranks[add_module] += delta_rank
            new_ranks[remove_module] -= delta_rank
            moves.append(RankMove(remove_module, add_module, delta_rank))
            break

    return RankAllocationResult(old_ranks=old_ranks, new_ranks=new_ranks, moves=moves, rejection_stats=rejection_stats)


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

    rejection = result.rejection_stats or {}
    lines.append(
        "[rank_allocation_lora] reject counters "
        f"add_max={rejection.get('add_max', 0)} "
        f"remove_min={rejection.get('remove_min', 0)} "
        f"tail_energy={rejection.get('tail_energy', 0)} "
        f"hysteresis={rejection.get('hysteresis', 0)} "
        f"same_module={rejection.get('same_module', 0)} "
        f"remove_used={rejection.get('remove_already_used', 0)}"
    )

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
