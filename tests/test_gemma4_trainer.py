"""Gemma 4 Tunix student: loader parity with transformers, LoRA export (q, fused k/v, o, MLP) and merge."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "trainer"))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tunix.models.gemma4")
pytest.importorskip("qwix")
if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
  pytest.skip("transformers without Gemma 4", allow_module_level=True)
from flax import nnx  # noqa: E402
from safetensors import safe_open  # noqa: E402

import student_models as sm  # noqa: E402
import trainer_core as tcore  # noqa: E402

LOGIT_ATOL = 1e-4  # float32 on both sides; see test_qwen3_trainer.py
VOCAB = 512

# A Gemma 4 E2B-shaped text config, scaled down, written in the released checkpoint's schema: per-layer
# inputs, sliding/full attention with a wider global head, KV sharing in the last layers, double-wide
# MLPs in the shared layers, proportional RoPE on full attention and a final logit soft-cap.
TEXT = {
    "model_type": "gemma4_text", "attention_bias": False, "attention_dropout": 0.0, "attention_k_eq_v": False,
    "enable_moe_block": False, "final_logit_softcapping": 30.0, "global_head_dim": 32, "head_dim": 16,
    "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 64, "hidden_size_per_layer_input": 8,
    "intermediate_size": 128, "layer_types": ["sliding_attention"] * 4 + ["full_attention"]
                                              + ["sliding_attention"] * 4 + ["full_attention"],
    "max_position_embeddings": 1024, "num_attention_heads": 4, "num_global_key_value_heads": None,
    "num_hidden_layers": 10, "num_key_value_heads": 1, "num_kv_shared_layers": 4, "rms_norm_eps": 1e-6,
    "rope_parameters": {"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                                           "rope_type": "proportional"},
                        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
    "sliding_window": 8, "tie_word_embeddings": True, "use_bidirectional_attention": None,
    "use_double_wide_mlp": True, "vocab_size": VOCAB, "vocab_size_per_layer_input": VOCAB,
    "pad_token_id": 0, "bos_token_id": 2, "eos_token_id": 1, "initializer_range": 0.2,
}
CONFIG = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "text_config": TEXT,
          "vision_config": None, "audio_config": None, "tie_word_embeddings": True, "eos_token_id": [1, 106]}


def _save_tiny(path: pathlib.Path, seed: int = 0):
  """A random tiny Gemma 4 in transformers (norm scales and layer scalars randomised too, so a scale
  convention mismatch cannot hide), saved with the released checkpoint's config.json."""
  torch.manual_seed(seed)
  cfg = transformers.Gemma4Config(**{k: v for k, v in CONFIG.items() if k != "architectures"})
  model = transformers.Gemma4ForConditionalGeneration(cfg).eval()
  g = torch.Generator().manual_seed(seed + 1)
  with torch.no_grad():
    for n, p in model.named_parameters():
      if "norm" in n or "layer_scalar" in n:
        p.copy_(1.0 + 0.3 * torch.randn(p.shape, generator=g))
  model.save_pretrained(path)
  (path / "config.json").write_text(json.dumps(CONFIG))
  # The released checkpoint also stores k/v projections and k-norm for the KV-shared layers (never
  # used in the forward pass); newer transformers versions do not write them. Add random ones.
  from safetensors.torch import load_file, save_file  # pylint: disable=g-import-not-at-top
  st = path / "model.safetensors"
  tensors = load_file(st)
  kv = TEXT["head_dim"]
  for i in range(TEXT["num_hidden_layers"] - TEXT["num_kv_shared_layers"], TEXT["num_hidden_layers"]):
    hd = TEXT["global_head_dim"] if TEXT["layer_types"][i] == "full_attention" else kv
    pre = f"model.language_model.layers.{i}.self_attn"
    for name in ("k_proj", "v_proj"):
      tensors[f"{pre}.{name}.weight"] = torch.randn(TEXT["num_key_value_heads"] * hd, TEXT["hidden_size"], generator=g)
    tensors[f"{pre}.k_norm.weight"] = 1.0 + 0.3 * torch.randn(hd, generator=g)
  save_file(tensors, st, metadata={"format": "pt"})
  return model


def _batch(seed=0):
  tokens = np.random.default_rng(seed).integers(3, VOCAB, (2, 40)).astype(np.int32)
  pad = np.ones_like(tokens)
  pad[1, 30:] = 0
  return tokens, pad


def _hf_logits(model, tokens, pad):
  with torch.no_grad():
    return model(input_ids=torch.tensor(tokens, dtype=torch.long), attention_mask=torch.tensor(pad)).logits.numpy()


def _tunix_logits(model, mesh, tokens, pad):
  with mesh:
    return np.asarray(sm.forward_logits(model, jnp.asarray(tokens), jnp.asarray(pad)))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
  d = tmp_path_factory.mktemp("gemma4")
  return d, _save_tiny(d)


def test_config_and_family(tiny):
  d, _ = tiny
  assert sm.model_family(str(d)) == "gemma4" and sm.gemma4_unshared_layers(str(d)) == 6
  cfg = sm.gemma4_config_from_hf(str(d), remat=False)
  assert cfg.global_key_size == 32 and cfg.override_kv_shared_ffw_hidden == 256 and cfg.final_logit_softcap == 30.0
  targets = sm.lora_targets(str(d))
  assert "layers/(0|1|2|3|4|5)/attn/kv_einsum" in targets


def test_tunix_logits_match_transformers(tiny):
  d, hf = tiny
  model = sm.load_gemma4(str(d), sm.make_mesh(1, 1), remat=False, flash=False, compute_dtype=jnp.float32)
  tokens, pad = _batch()
  got, ref = _tunix_logits(model, sm.make_mesh(1, 1), tokens, pad), _hf_logits(hf, tokens, pad)
  real = pad.astype(bool)
  assert np.abs(ref[real]).max() > 1.0
  assert np.abs(got - ref)[real].max() < LOGIT_ATOL


@pytest.mark.parametrize("field,value", [("enable_moe_block", True), ("attention_k_eq_v", True),
                                         ("hidden_activation", "silu"), ("attention_bias", True)])
def test_config_refuses_fields_tunix_cannot_express(tmp_path, field, value):
  cfg = json.loads(json.dumps(CONFIG))
  cfg["text_config"][field] = value
  (tmp_path / "config.json").write_text(json.dumps(cfg))
  with pytest.raises(ValueError):
    sm.gemma4_config_from_hf(str(tmp_path))


def test_merged_lora_export_reproduces_the_lora_model(tiny, tmp_path):
  d, _ = tiny
  mesh = sm.make_mesh(1, 1)
  model = sm.load_gemma4(str(d), mesh, remat=False, flash=False, compute_dtype=jnp.float32)
  lora = sm.apply_lora(model, mesh, sm.lora_targets(str(d)), 4, 8.0)
  rng = np.random.default_rng(3)
  for _, v in nnx.iter_graph(lora):
    if isinstance(v, nnx.LoRAParam):
      v.set_value(jnp.asarray(rng.normal(0, 0.05, v.get_value().shape), jnp.float32))
  tokens, pad = _batch(1)
  want = _tunix_logits(lora, mesh, tokens, pad)
  deltas = sm.lora_deltas_hf(lora, 8.0, 4, "gemma4")
  # q, k, v, o, gate, up, down in the 6 unshared layers; no k/v in the 4 KV-shared layers.
  assert len(deltas) == 6 * 7 + 4 * 5
  assert not any(f"layers.{i}.self_attn.{p}_proj" in k for k in deltas for i in range(6, 10) for p in ("k", "v"))
  out = tmp_path / "merged"
  manifest = tcore.merge_lora_into_hf_shards(str(d), str(out), deltas)
  assert manifest["n_modified_tensors"] == len(deltas)
  with safe_open(str(d / "model.safetensors"), "np") as a, safe_open(str(out / "model.safetensors"), "np") as b:
    untouched = [k for k in a.keys() if k not in deltas]
    assert untouched and all(np.array_equal(a.get_tensor(k), b.get_tensor(k)) for k in untouched)
  hf = transformers.Gemma4ForConditionalGeneration.from_pretrained(out, dtype=torch.float32).eval()
  got = _hf_logits(hf, tokens, pad)
  real = pad.astype(bool)
  assert np.abs(got - want)[real].max() < 1e-3
  base = _hf_logits(transformers.Gemma4ForConditionalGeneration.from_pretrained(d, dtype=torch.float32).eval(),
                    tokens, pad)
  assert np.abs(base - want)[real].max() > 1e-2  # the LoRA changed the model, so the match is meaningful


def _gemma_tokenizer():
  try:
    return transformers.AutoTokenizer.from_pretrained("google/gemma-4-E2B-it", local_files_only=True)
  except (OSError, ValueError):
    pytest.skip("Gemma 4 tokenizer not in the local Hugging Face cache")


def test_only_the_response_and_turn_close_are_trained_and_the_prompt_is_the_serving_prompt():
  import data as dc  # pylint: disable=g-import-not-at-top

  tok = _gemma_tokenizer()
  msgs = [{"role": "system", "content": "You are an expert mathematician."},
          {"role": "user", "content": "Problem: 1+1?\n\nReason step by step, briefly."}]
  rec = dc.Record.from_dict({"id": "train-0#s0", "messages": msgs, "response": "1+1=2.\n\\boxed{2}"})
  r = dc.render_example(rec, tok, max_seq_len=256)
  ids = r.input_ids[: r.n_real]
  assert tok.decode(ids[r.loss_weight[: r.n_real] > 0]) == "1+1=2.\n\\boxed{2}<turn|>"
  serving = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
  assert tok.decode(ids).startswith(serving) and serving.endswith("<|turn>model\n")


def test_truncated_grpo_record_leaves_turn_close_out_of_the_loss():
  import data as dc  # pylint: disable=g-import-not-at-top

  tok = _gemma_tokenizer()
  msgs = [{"role": "system", "content": "You are an expert mathematician."},
          {"role": "user", "content": "Problem: 1+1?\n\nReason step by step, briefly."}]
  rec = dc.Record.from_dict({"id": "train-0#s0", "messages": msgs, "response": "1+1 is, let me think, 2 and",
                             "reward": 0.0, "truncated": True})
  r = dc.render_example(rec, tok, max_seq_len=256)
  assert tok.decode(r.input_ids[: r.n_real][r.loss_weight[: r.n_real] > 0]) == "1+1 is, let me think, 2 and"
