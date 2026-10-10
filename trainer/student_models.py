"""Build the dense Qwen3 student on a TPU mesh (Tunix), apply LoRA, and export LoRA deltas in HF layout.

The loader reads bf16 HF safetensors directly and verifies every architecture field against
config.json before any weight is touched.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

# qwix LoRA regexes are matched with re.fullmatch against '/'-joined nnx module paths.
# Tunix dense Qwen3: attention projections are `Einsum` modules (layers/N/attn/{q,k,v,o}_proj) and the
# MLP projections are `nnx.Linear` modules (layers/N/mlp/{gate,up,down}_proj); qwix wraps both.
LORA_TARGETS = r".*/attn/(q_proj|k_proj|v_proj|o_proj)|.*/mlp/(gate_proj|up_proj|down_proj)"
# HF tensors a LoRA export may modify, one per adapted module.
LORA_HF_KEY = (r"model\.layers\.[0-9]+\.(self_attn\.(q|k|v|o)_proj"
               r"|mlp\.(gate|up|down)_proj)\.weight")


def maybe_init_distributed() -> None:
  import jax  # pylint: disable=g-import-not-at-top

  hosts = os.environ.get("TPU_WORKER_HOSTNAMES", "")
  if "," in hosts:
    jax.distributed.initialize()


def read_hf_config(hf_dir: str) -> dict[str, Any]:
  with open(os.path.join(hf_dir, "config.json")) as f:
    return json.load(f)


def make_mesh(fsdp: int, tp: int) -> Any:
  import jax  # pylint: disable=g-import-not-at-top

  # Auto axes: Tunix models rely on with_sharding_constraint hints, which become hard asserts
  # under jax>=0.11's default Explicit axes.
  return jax.make_mesh((fsdp, tp), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)


def _bind_rope_theta(rope_theta: int) -> None:
  """Make Tunix Qwen3 attention use the checkpoint's RoPE base.

  Tunix's attention calls `apply_rope(x, positions, head_dim=...)` without passing
  `ModelConfig.rope_theta`, so it always rotates with the function default (1e6).
  Some Qwen3 checkpoints use another base (e.g. 5e6). The module-level `apply_rope` is rebound to
  the original function with `rope_theta` fixed; the attention looks it up at call time.
  """
  import functools  # pylint: disable=g-import-not-at-top
  from tunix.models.qwen3 import model as model_lib  # pylint: disable=g-import-not-at-top

  orig = getattr(model_lib, "_apply_rope_unbound", model_lib.apply_rope)
  model_lib._apply_rope_unbound = orig  # pylint: disable=protected-access
  model_lib.apply_rope = functools.partial(orig, rope_theta=rope_theta)


def _hf_rope_theta(c: dict[str, Any]) -> int:
  """RoPE base from config.json: top-level `rope_theta` (transformers 4.x) or `rope_parameters`
  (transformers >= 5). Only the default (unscaled) RoPE is accepted."""
  params = c.get("rope_parameters")
  if params is not None:
    if params.get("rope_type", "default") != "default" or set(params) - {"rope_theta", "rope_type"}:
      raise ValueError(f"rope_parameters Tunix Qwen3 cannot express: {params}")
    theta = params["rope_theta"]
  else:
    theta = c["rope_theta"]
  if float(theta) != int(theta):
    raise ValueError(f"rope_theta={theta} is not an integer")
  return int(theta)


def qwen3_config_from_hf(hf_dir: str, remat: Any = True, flash: bool = False, compute_dtype: Any = None) -> Any:
  """Tunix ModelConfig for a dense Qwen3 checkpoint (model_type "qwen3"), built from config.json.

  Every field Tunix cannot express is asserted to be at its Tunix-implied value; the RoPE base
  and tied embeddings come from config.json, not from a preset.
  """
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from tunix.models.qwen3 import model as model_lib  # pylint: disable=g-import-not-at-top

  c = read_hf_config(hf_dir)
  if c.get("model_type") != "qwen3":
    raise ValueError(f"expected qwen3, got {c.get('model_type')}")
  if "quantization_config" in c:
    raise ValueError(f"{hf_dir} is quantized; use the bf16 checkpoint")
  implied = {
      "attention_bias": False,
      "rope_scaling": None,
      "use_sliding_window": False,
      "hidden_act": "silu",
  }
  bad = {k: (c.get(k), v) for k, v in implied.items() if c.get(k, v) != v}
  layer_types = c.get("layer_types")
  if layer_types is not None and set(layer_types) != {"full_attention"}:
    bad["layer_types"] = (sorted(set(layer_types)), ["full_attention"])
  if bad:
    raise ValueError(f"config.json fields Tunix Qwen3 cannot express: {bad}")
  return model_lib.ModelConfig(
      num_layers=c["num_hidden_layers"],
      vocab_size=c["vocab_size"],
      embed_dim=c["hidden_size"],
      hidden_dim=c["intermediate_size"],
      num_heads=c["num_attention_heads"],
      head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
      num_kv_heads=c["num_key_value_heads"],
      rope_theta=_hf_rope_theta(c),
      norm_eps=c["rms_norm_eps"],
      use_tied_embedding=bool(c.get("tie_word_embeddings", False)),
      num_experts=None,
      num_experts_per_tok=None,
      # remat: bool (True = attention-block remat) or a RematConfig name: NONE | BLOCK | DECODER.
      remat_config=(getattr(model_lib.RematConfig, remat) if isinstance(remat, str)
                    else model_lib.RematConfig.BLOCK if remat else model_lib.RematConfig.NONE),
      dtype=compute_dtype or jnp.bfloat16,
      param_dtype=compute_dtype or jnp.bfloat16,
      use_flash_attention=flash,
      flash_attention_block_size=1024,
  )


def load_qwen3(hf_dir: str, mesh: Any, remat: Any = True, flash: bool = False, compute_dtype: Any = None) -> Any:
  """Tunix dense Qwen3 from HF safetensors, placed on `mesh`."""
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from tunix.models import safetensors_loader  # pylint: disable=g-import-not-at-top
  from tunix.models.qwen3 import model as model_lib  # pylint: disable=g-import-not-at-top
  from tunix.models.qwen3 import params as params_lib  # pylint: disable=g-import-not-at-top

  cfg = qwen3_config_from_hf(hf_dir, remat=remat, flash=flash, compute_dtype=compute_dtype)
  _bind_rope_theta(cfg.rope_theta)
  print(f"[qwen3] layers={cfg.num_layers} embed={cfg.embed_dim} mlp={cfg.hidden_dim} heads={cfg.num_heads}/"
        f"{cfg.num_kv_heads} rope_theta={cfg.rope_theta} tied={cfg.use_tied_embedding}", flush=True)
  with mesh:
    model = safetensors_loader.load_and_create_model(
        file_dir=hf_dir,
        model_class=model_lib.Qwen3,
        config=cfg,
        key_mapping=params_lib._get_key_and_transform_mapping,  # pylint: disable=protected-access
        mesh=mesh,
        dtype=compute_dtype or jnp.bfloat16,
    )
  return model


def lora_deltas_hf(lora_model: Any, alpha: float, rank: int) -> dict[str, np.ndarray]:
  """Tunix LoRA -> HF-keyed float32 deltas ([out, in]), scaled by alpha/rank, for attention and MLP LoRA.

  Same reshape convention as tunix.models.safetensors_saver, but multi-host safe and returned
  for the sharded writer (Tunix's saver only reads a single model.safetensors). An `nnx.Linear`
  kernel [in, out] reshapes like the q/k/v einsum weights; every key is checked against the
  LoRA target set.
  """
  import re  # pylint: disable=g-import-not-at-top

  from flax import nnx  # pylint: disable=g-import-not-at-top
  from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top

  pairs: dict[str, dict[str, Any]] = {}
  for path, var in nnx.iter_graph(lora_model):
    if isinstance(var, nnx.LoRAParam):
      layer = ".".join(str(p) for p in path[:-1])
      pairs.setdefault(layer, {})[str(path[-1]).rsplit("_", 1)[-1]] = var
  out = {}
  for layer, ab in pairs.items():
    a = np.asarray(multihost_utils.process_allgather(ab["a"].get_value(), tiled=True), np.float32)
    b = np.asarray(multihost_utils.process_allgather(ab["b"].get_value(), tiled=True), np.float32)
    delta = np.tensordot(a, b, axes=([a.ndim - 1], [0])) * (alpha / rank)
    # q/k/v: w[D, N, H] -> HF [N*H, D];  o: w[N, H, D] -> HF [D, N*H]
    if layer.endswith("o_proj"):
      d2 = delta.reshape(-1, delta.shape[-1])
    else:
      d2 = delta.reshape(delta.shape[0], -1)
    key = f"model.{layer}.weight".replace(".attn.", ".self_attn.")
    out[key] = np.ascontiguousarray(d2.T)
  bad = sorted(k for k in out if not re.fullmatch(LORA_HF_KEY, k))
  if bad:
    raise ValueError(f"LoRA deltas outside the LoRA targets: {bad[:5]}")
  return out


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------


def _shard_rank_axis(sharding: Any, shape: tuple[int, ...], rank: int) -> Any:
  """Also shard a LoRA a/b param's rank dimension over the mesh axes its annotation leaves unused.

  The base kernel's annotation shards a LoRA param along one mesh axis only, which keeps a full
  copy per device along every other axis — for the params, their gradients and both Adam moments.
  """
  from jax.sharding import NamedSharding, PartitionSpec  # pylint: disable=g-import-not-at-top

  spec = list(sharding.spec) + [None] * (len(shape) - len(sharding.spec))
  used = {a for s in spec if s is not None for a in (s if isinstance(s, tuple) else (s,))}
  free = tuple(a for a, n in sharding.mesh.shape.items() if n > 1 and a not in used)
  size = 1
  for a in free:
    size *= sharding.mesh.shape[a]
  dims = [i for i, (n, s) in enumerate(zip(shape, spec)) if n == rank and s is None]
  if not free or not dims or rank % size:
    return sharding
  spec[dims[-1]] = free if len(free) > 1 else free[0]
  return NamedSharding(sharding.mesh, PartitionSpec(*spec))


def apply_lora(model: Any, mesh: Any, targets: str, rank: int, alpha: float, seed: int = 0) -> Any:
  """qwix LoRA on the modules matching `targets`; LoRA params in float32, placed on the mesh."""
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from flax import nnx  # pylint: disable=g-import-not-at-top
  import qwix  # pylint: disable=g-import-not-at-top
  from tunix.utils import compat  # pylint: disable=g-import-not-at-top

  provider = qwix.LoraProvider(module_path=targets, rank=rank, alpha=alpha)
  model_input = model.get_model_input() if hasattr(model, "get_model_input") else {}
  # The dummy input has batch 2 (sized for a 2-device axis). Inside an active mesh the model's
  # activation sharding constraints split the batch over fsdp, so repeat it to the device count.
  n_dev = int(np.prod(list(mesh.shape.values())))
  batches = {v.shape[0] for v in model_input.values() if hasattr(v, "shape") and len(v.shape) >= 1}
  if len(batches) == 1 and n_dev % next(iter(batches)) == 0:
    reps = n_dev // next(iter(batches))
    model_input = {k: (jnp.repeat(v, reps, axis=0) if hasattr(v, "shape") and len(v.shape) >= 1 else v)
                   for k, v in model_input.items()}
  # qwix finds the target modules by tracing one forward pass; Tunix block remat wraps each
  # block in jax.checkpoint, which hides those calls from qwix. Trace without remat, then
  # restore the configured remat for training.
  cfgs = {id(c): c for _, v in nnx.iter_graph(model) if (c := getattr(v, "config", None)) is not None
          and hasattr(c, "remat_config")}
  saved = {k: c.remat_config for k, c in cfgs.items()}
  if cfgs:
    from tunix.models.qwen3 import model as qwen3_lib  # pylint: disable=g-import-not-at-top
    for c in cfgs.values():
      c.remat_config = qwen3_lib.RematConfig.NONE
  try:
    lora_model = qwix.apply_lora_to_model(model, provider, rngs=nnx.Rngs(seed), **model_input)
    nnx.pop(lora_model, nnx.Intermediate)  # values sown during the tracing forward pass
  finally:
    for k, c in cfgs.items():
      c.remat_config = saved[k]
  for _, v in nnx.iter_graph(lora_model):
    c = getattr(v, "config", None)
    if c is not None and hasattr(c, "remat_config") and id(c) not in saved:
      c.remat_config = next(iter(saved.values()))
  # Place every param (base + fresh LoRA a/b) on its annotated NamedSharding. device_put
  # rather than with_sharding_constraint: qwix creates LoRA params eagerly on one device,
  # and an eager sharding constraint rejects SingleDeviceSharding inputs under jax>=0.11.
  from jax.sharding import NamedSharding, PartitionSpec  # pylint: disable=g-import-not-at-top

  with compat.set_mesh(mesh):
    state = nnx.state(lora_model)
    specs = nnx.get_partition_spec(state)
    # A param without a sharding annotation has spec None, which device_put reads as "default
    # device" (an error for a multi-host array). Keep its current sharding when it already spans
    # the mesh; otherwise replicate it.
    flat_specs, treedef = jax.tree_util.tree_flatten(specs, is_leaf=lambda x: x is None or isinstance(x, PartitionSpec))
    flat_state = treedef.flatten_up_to(state)
    unannotated = [i for i, s in enumerate(flat_specs) if not isinstance(s, PartitionSpec)]
    keep = lambda x: x.sharding if len(x.sharding.device_set) > 1 else NamedSharding(mesh, PartitionSpec())
    flat_shardings = [NamedSharding(mesh, s) if isinstance(s, PartitionSpec) else jax.tree.map(keep, a)
                      for s, a in zip(flat_specs, flat_state)]
    if unannotated:
      paths = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(
          specs, is_leaf=lambda x: x is None or isinstance(x, PartitionSpec))[0]]
      print(f"[lora] {len(unannotated)} state entries without a sharding annotation keep or replicate their "
            f"placement, e.g. {[paths[i][-60:] for i in unannotated[:3]]}", flush=True)
    nnx.update(lora_model, jax.device_put(state, treedef.unflatten(flat_shardings)))
    lora_state = nnx.state(lora_model, nnx.LoRAParam)
    nnx.update(lora_model, jax.device_put(lora_state, jax.tree.map(
        lambda x: _shard_rank_axis(x.sharding, x.shape, rank), lora_state)))
    # LoRA params (and so the optimizer state) live in float32. qwix creates them in the base
    # kernel's dtype; in bf16, AdamW's beta2 = 0.999 rounds to 1.0, so the second moment stays
    # 0, the bias correction 1 - beta2^t is 0, and the first update is 0/0 = NaN.
    lora_state = nnx.state(lora_model, nnx.LoRAParam)
    nnx.update(lora_model, jax.tree.map(lambda x: x.astype(jnp.float32), lora_state))
    # The optimizer state takes its sharding from the params' metadata, not from their placement.
    # qwix copies the base kernel's annotation (one mesh axis); record the actual placement instead.
    for _, v in nnx.iter_graph(lora_model):
      if isinstance(v, nnx.LoRAParam):
        v.set_metadata(out_sharding=tuple(v.value.sharding.spec), sharding_rules=())
  n = sum(1 for _, v in nnx.iter_graph(lora_model) if isinstance(v, nnx.LoRAParam))
  if n == 0:
    raise RuntimeError(f"LoRA regex {targets!r} matched no module")
  print(f"[lora] {n // 2} adapted modules, rank={rank}, alpha={alpha}", flush=True)
  return lora_model


def lm_head_f32(model: Any, h: Any) -> Any:
  """Tunix Qwen3 LM head with float32 output.

  Tunix's head is a bf16 einsum whose bf16 output is cast to float32 afterwards, which
  rounds every logit to bf16 (a 0.125 step at |logit| in [16, 32)). Here the same bf16
  weights and activations are multiplied with a float32 result.
  """
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

  dt = model.config.dtype
  x = jnp.astype(h, dt)
  if model.config.use_tied_embedding:
    w = jnp.astype(model.embedder.input_embedding.value, dt)
    return jnp.einsum("BTD,VD->BTV", x, w, preferred_element_type=jnp.float32)
  w = jnp.astype(model.lm_head.w.value, dt)
  return jnp.einsum(model.lm_head.einsum_str, x, w, preferred_element_type=jnp.float32)


def forward_logits(model: Any, tokens: Any, pad_mask: Any) -> Any:
  """Full-vocab float32 logits [B, L, V] for right-padded tokens (dense causal mask path)."""
  from tunix.sft import utils as sft_utils  # pylint: disable=g-import-not-at-top

  positions = sft_utils.build_positions_from_mask(pad_mask)
  attn = sft_utils.make_causal_attn_mask(pad_mask)
  h, _ = model(tokens, positions, None, attn, segment_ids=pad_mask.astype("int32"), skip_lm_head=True)
  return lm_head_f32(model, h).astype("float32")
