import ast
import importlib.util
import pathlib
import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "para_eff_pt" / "pt_rank_allocation_lora" / "rank_allocation_lora.py"


def load_rank_allocation_module():
    spec = importlib.util.spec_from_file_location("rank_allocation_lora", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RankAllocationLoRaSmokeTest(unittest.TestCase):
    def test_module_defines_expected_public_api(self):
        source = MODULE_PATH.read_text()
        tree = ast.parse(source)
        class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}

        self.assertIn("RankAllocationLoRaModel", class_names)
        self.assertIn("RankAllocationLoRaLinear", class_names)
        self.assertIn("allocate_rank_budget", function_names)
        self.assertIn("summarize_rank_allocation", function_names)

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_probe_branch_is_zero_output_initially(self):
        module = load_rank_allocation_module()
        layer = module.RankAllocationLoRaLinear(4, 3, 2, lora_alpha=2, bias=True)
        x = torch.randn(5, 4)

        before = layer(x).detach()
        layer.enable_probe(probe_rank=1, sigma=1e-3)
        after = layer(x).detach()

        self.assertTrue(torch.allclose(before, after, atol=0, rtol=0))

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_refactorization_preserves_output_for_same_rank_and_changes_shapes(self):
        module = load_rank_allocation_module()
        torch.manual_seed(0)
        layer = module.RankAllocationLoRaLinear(4, 3, 2, lora_alpha=2, bias=False)
        x = torch.randn(5, 4)

        before = layer(x).detach()
        layer.refactorize_with_rank(2)
        same_rank = layer(x).detach()
        layer.refactorize_with_rank(1)

        self.assertTrue(torch.allclose(before, same_rank, atol=2e-5, rtol=2e-5))
        self.assertEqual(tuple(layer.lora_A.shape), (1, 4))
        self.assertEqual(tuple(layer.lora_B.shape), (3, 1))
        self.assertEqual(layer.rank, 1)

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_allocator_moves_rank_from_low_score_to_high_score_with_budget_fixed(self):
        module = load_rank_allocation_module()
        low = module.RankAllocationLoRaLinear(8, 8, 4, bias=False)
        high = module.RankAllocationLoRaLinear(8, 8, 4, bias=False)
        low.module_name = "mlp.down_proj"
        high.module_name = "attn.q_proj"
        low.score = 0.1
        high.score = 10.0

        result = module.allocate_rank_budget(
            [low, high],
            delta_rank=2,
            top_k=1,
            min_ratio=0.25,
            max_ratio=0.75,
            hysteresis=0.1,
        )

        self.assertEqual(result.new_ranks[low], 2)
        self.assertEqual(result.new_ranks[high], 6)
        self.assertEqual(sum(result.old_ranks.values()), sum(result.new_ranks.values()))
        self.assertEqual(len(result.moves), 1)

    @unittest.skipIf(torch is None, "torch is not installed in this Python environment")
    def test_model_replaces_target_linears(self):
        module = load_rank_allocation_module()

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Linear(4, 4, bias=False)
                self.other = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                return self.other(self.attn(x))

        wrapped = module.RankAllocationLoRaModel(TinyModel(), target_modules=["attn"], r=2)

        self.assertIsInstance(wrapped.wrapped_model.attn, module.RankAllocationLoRaLinear)
        self.assertIsInstance(wrapped.wrapped_model.other, nn.Linear)


if __name__ == "__main__":
    unittest.main()
