#!/usr/bin/env python3
"""JAX-trainer vs vLLM next-token parity on 16 fixed prompts.

JAX side : the trainer's Tunix student (Qwen3 or Gemma 4), bf16 weights from HF safetensors.
vLLM side: POST {endpoint}/v1/completions with the prompt as token ids, max_tokens=1,
           echo=true, prompt_logprobs=K. The vLLM server under test MUST serve the same
           bf16 checkpoint with --kv-cache-dtype auto (FP8 weights or FP8 KV make the
           comparison meaningless).

Per prompt position t >= 1 (the distribution of token t given tokens < t):
  top1        : vLLM rank-1 token == argmax of JAX logits at t-1
  kl          : KL(P_vllm || Q_jax) over K+1 bins (vLLM top-K ids + "rest"). Coarsening
                can only lower KL (data processing inequality), so this is a lower bound
                on the full-vocabulary KL; it is exact up to the rest-bin mass.
  dlogp_prompt: |logp_vllm(x_t) - logp_jax(x_t)| on the actual prompt token
  dlogp_topk  : max over the K vLLM ids of |logp_vllm - logp_jax|
Gate: exit 1 if mean top1 < 0.99 or mean kl >= 1e-3. All numbers are written as computed.

Multi-host: every process builds the model and runs the forward; process 0 alone calls
vLLM and writes the JSON. vLLM results are broadcast so all hosts use identical ids.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import data as dc  # noqa: E402  pylint: disable=g-import-not-at-top
import student_models as sm  # noqa: E402  pylint: disable=g-import-not-at-top

K = 20  # vLLM --max-logprobs default
TOP1_MIN = 0.99
KL_MAX = 1e-3

_SCHEMA_BLOCK = (
    "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT, signup_date DATE);\n"
    "/* 3 example rows from customers:\ncustomer_id | name | city | signup_date\n"
    "1 | Ann Lee | Oslo | 2021-03-04\n2 | Bo Chen | Rome | 2022-07-19\n3 | Cy Diaz | Lima | 2020-11-30\n*/\n"
    "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id), "
    "amount REAL, status TEXT, order_date DATE);\n"
    "/* 3 example rows from orders:\norder_id | customer_id | amount | status | order_date\n"
    "10 | 1 | 25.5 | shipped | 2023-01-02\n11 | 2 | 7.0 | pending | 2023-01-05\n12 | 1 | 99.9 | shipped | 2023-02-11\n*/\n"
)
# Fixed, versioned prompt set. Changing it changes the parity result's meaning.
PROMPTS: list[list[dict[str, str]]] = [
    [{"role": "user", "content": "What is the capital of Australia? Answer in one word."}],
    [{"role": "user", "content": "Write a bash one-liner that counts the lines in every .py file under src/."}],
    [{"role": "user", "content": "Explain the difference between a process and a thread in two sentences."}],
    [{"role": "user", "content": "Translate to French: The build failed because the linker could not find libssl."}],
    [{"role": "user", "content": "def fib(n):\n    # complete this function iteratively\n"}],
    [{"role": "user", "content": "List three causes of a 'Permission denied (publickey)' error from ssh."}],
    [{"role": "user", "content": "Compute 17 * 23 and show the steps."}],
    [{"role": "user", "content": "Give a regex that matches an IPv4 address and explain each part."}],
    [{"role": "system", "content": "You are a terse assistant."},
     {"role": "user", "content": "How do I undo the last git commit but keep the changes?"}],
    [{"role": "user", "content": "Summarise what `tar -xzvf archive.tar.gz -C /opt` does."},
     {"role": "assistant", "content": "It extracts a gzip-compressed tar archive verbosely into /opt."},
     {"role": "user", "content": "And how do I list the contents without extracting?"}],
    [{"role": "user", "content": "Write a Python function that parses a CSV line with quoted fields."}],
    [{"role": "user", "content": "Why might `pip install` succeed but `import foo` fail in the same shell?"}],
    # Long prompts: exercise the long-context attention path.
    [{"role": "system", "content": "You are an expert data analyst who writes correct SQLite queries."},
     {"role": "user", "content": "Database schema (SQLite):\n" + _SCHEMA_BLOCK * 8 +
      "\nQuestion: Which city has the highest total shipped order amount?"}],
    [{"role": "user", "content": "Schema:\n" + _SCHEMA_BLOCK * 12 + "\nWrite a query listing customers with no orders."}],
    [{"role": "user", "content": "Schema:\n" + _SCHEMA_BLOCK * 10 + "\nCount pending orders per customer."},
     {"role": "assistant", "content": "```sql\nSELECT customer_id, COUNT(*) FROM orders WHERE status = 'pending' "
      "GROUP BY customer_id\n```"},
     {"role": "user", "content": "Now include customers with zero pending orders.\n" + _SCHEMA_BLOCK * 4}],
    [{"role": "user", "content": "Read this schema and name the foreign key:\n" + _SCHEMA_BLOCK * 15}],
]
assert len(PROMPTS) == 16


def _bucket(n: int, base: int = dc.MXU) -> int:
  """Smallest base * 2^j >= n. `base` must divide every padded length the kernel sees."""
  b = base
  while b < n:
    b *= 2
  return b


def query_vllm(endpoint: str, served_model: str, ids: list[int], timeout: float) -> dict:
  import requests  # pylint: disable=g-import-not-at-top

  # Only prompt_logprobs is used (it always includes the actual prompt token). The echoed
  # completion logprobs are not requested: vLLM's echo formatting can raise KeyError when a
  # prompt token is absent from its top-K dict.
  body = {
      "model": served_model,
      "prompt": ids,
      "max_tokens": 1,
      "temperature": 0.0,
      "prompt_logprobs": K,
  }
  r = requests.post(f"{endpoint.rstrip('/')}/v1/completions", json=body, timeout=timeout)
  r.raise_for_status()
  return r.json()


def parse_prompt_logprobs(resp: dict, ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """-> topk_ids [L, K] (int32, -1 = absent), topk_lp [L, K] f32, tok_lp [L] f32 (nan at t=0)."""
  plp = resp["choices"][0].get("prompt_logprobs")
  if plp is None:
    raise RuntimeError("vLLM response has no prompt_logprobs; server too old or flag rejected")
  L = len(ids)
  if len(plp) < L:
    raise RuntimeError(f"prompt_logprobs has {len(plp)} entries for {L} prompt tokens")
  top_ids = np.full((L, K), -1, np.int32)
  top_lp = np.full((L, K), np.nan, np.float32)
  tok_lp = np.full((L,), np.nan, np.float32)
  for t in range(1, L):
    entry = plp[t]
    items = [(int(k), v) for k, v in entry.items()]
    ranked = sorted((it for it in items if it[1]["rank"] <= K), key=lambda it: it[1]["rank"])
    for j, (tid, v) in enumerate(ranked[:K]):
      top_ids[t, j] = tid
      top_lp[t, j] = v["logprob"]
    actual = entry.get(str(ids[t]))
    if actual is None:
      raise RuntimeError(f"position {t}: actual prompt token {ids[t]} missing from prompt_logprobs")
    tok_lp[t] = actual["logprob"]
  return top_ids, top_lp, tok_lp


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--hf_dir", required=True, help="local bf16 HF checkpoint dir (the one vLLM serves)")
  ap.add_argument("--endpoint", required=True, help="e.g. http://localhost:8100")
  ap.add_argument("--served_model", required=True)
  ap.add_argument("--fsdp", type=int, default=2)
  ap.add_argument("--tp", type=int, default=4)
  ap.add_argument("--timeout", type=float, default=600.0)
  ap.add_argument("--flash", action="store_true", help="use the splash kernel used in training")
  ap.add_argument("--out", required=True)
  ap.add_argument("--dump_npz", default="", help="also save raw per-position arrays (vLLM top-K, JAX log-probs)")
  args = ap.parse_args()

  sm.maybe_init_distributed()
  import jax  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
  from jax.experimental import multihost_utils  # pylint: disable=g-import-not-at-top
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  tok = AutoTokenizer.from_pretrained(args.hf_dir)
  texts = [tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in PROMPTS]
  all_ids = [list(map(int, tok(t, add_special_tokens=False)["input_ids"])) for t in texts]
  pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

  # ---- vLLM (process 0), then broadcast ------------------------------------------------
  vl = []
  for ids in all_ids:
    L = len(ids)
    if jax.process_index() == 0:
      top_ids, top_lp, tok_lp = parse_prompt_logprobs(query_vllm(args.endpoint, args.served_model, ids, args.timeout), ids)
    else:
      top_ids = np.zeros((L, K), np.int32)
      top_lp = np.zeros((L, K), np.float32)
      tok_lp = np.zeros((L,), np.float32)
    top_ids, top_lp, tok_lp = multihost_utils.broadcast_one_to_all((top_ids, top_lp, tok_lp))
    vl.append((np.asarray(top_ids), np.asarray(top_lp), np.asarray(tok_lp)))

  # ---- JAX model -----------------------------------------------------------------------
  t0 = time.time()
  mesh = sm.make_mesh(args.fsdp, args.tp)
  model = sm.load_student(args.hf_dir, mesh, remat=False, flash=args.flash)
  # The splash kernel needs the padded length to be a multiple of its block size.
  bucket_base = max(dc.MXU, model.config.flash_attention_block_size) if args.flash else dc.MXU
  load_s = time.time() - t0
  from flax import nnx  # pylint: disable=g-import-not-at-top

  @nnx.jit
  def jax_logprobs(m, tokens, pad_mask, gather_ids):
    logits = sm.forward_logits(m, tokens, pad_mask)
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, gather_ids, axis=-1), jnp.argmax(logits, axis=-1)

  # Pair prompts of similar length (batch 2 = fsdp) and pad each pair to a 256-multiple bucket.
  order = sorted(range(len(all_ids)), key=lambda i: len(all_ids[i]))
  per_prompt = {}
  with mesh:
    for a in range(0, len(order), args.fsdp):
      grp = order[a : a + args.fsdp]
      Lp = _bucket(max(len(all_ids[i]) for i in grp), base=bucket_base)
      tokens = np.full((len(grp), Lp), pad_id, np.int32)
      mask = np.zeros((len(grp), Lp), np.int32)
      # gather[b, s] = [vLLM top-K ids at position s+1, actual token s+1]; logits at s predict s+1.
      gather = np.zeros((len(grp), Lp, K + 1), np.int32)
      for b, i in enumerate(grp):
        ids = all_ids[i]
        L = len(ids)
        tokens[b, :L] = ids
        mask[b, :L] = 1
        top_ids = vl[i][0]
        g = np.where(top_ids[1:] >= 0, top_ids[1:], 0)
        gather[b, : L - 1, :K] = g
        gather[b, : L - 1, K] = ids[1:]
      lp, am = jax_logprobs(model, jnp.asarray(tokens), jnp.asarray(mask), jnp.asarray(gather))
      lp = np.asarray(multihost_utils.process_allgather(lp, tiled=True))
      am = np.asarray(multihost_utils.process_allgather(am, tiled=True))
      for b, i in enumerate(grp):
        per_prompt[i] = (lp[b], am[b])

  if jax.process_index() != 0:
    return 0

  # ---- metrics (process 0) -------------------------------------------------------------
  rows, top1_all, kl_all, dprompt_all, dtopk_all = [], [], [], [], []
  n_rest_nonpositive = 0
  for i, ids in enumerate(all_ids):
    L = len(ids)
    top_ids, top_lp, tok_lp = vl[i]
    jlp, jam = per_prompt[i]
    agree, kls, dps, dks = [], [], [], []
    for t in range(1, L):
      valid = top_ids[t] >= 0
      v_lp = top_lp[t][valid].astype(np.float64)
      j_lp = jlp[t - 1, :K][valid].astype(np.float64)
      agree.append(int(top_ids[t, 0] == jam[t - 1]))
      p = np.exp(v_lp)
      q = np.exp(j_lp)
      p_rest, q_rest = 1.0 - p.sum(), 1.0 - q.sum()
      kl = float(np.sum(p * (v_lp - j_lp)))
      if p_rest > 0 and q_rest > 0:
        kl += p_rest * (math.log(p_rest) - math.log(q_rest))
      else:
        n_rest_nonpositive += 1
      kls.append(kl)
      dps.append(abs(float(tok_lp[t]) - float(jlp[t - 1, K])))
      dks.append(float(np.max(np.abs(v_lp - j_lp))))
    rows.append({
        "prompt_index": i,
        "n_tokens": L,
        "n_scored_positions": L - 1,
        "top1_agreement": float(np.mean(agree)),
        "mean_kl": float(np.mean(kls)),
        "max_kl": float(np.max(kls)),
        "max_abs_dlogprob_prompt_token": float(np.max(dps)),
        "max_abs_dlogprob_topk": float(np.max(dks)),
        "n_top1_disagree": int(len(agree) - sum(agree)),
    })
    top1_all += agree
    kl_all += kls
    dprompt_all += dps
    dtopk_all += dks

  summary = {
      "top1_agreement": float(np.mean(top1_all)),
      "mean_kl": float(np.mean(kl_all)),
      "max_abs_dlogprob_prompt_token": float(np.max(dprompt_all)),
      "max_abs_dlogprob_topk": float(np.max(dtopk_all)),
      "n_positions": len(top1_all),
      "n_rest_bin_nonpositive": n_rest_nonpositive,
  }
  passed = summary["top1_agreement"] >= TOP1_MIN and summary["mean_kl"] < KL_MAX
  out = {
      "hf_dir": os.path.abspath(args.hf_dir),
      "hf_config_sha256": dc.sha256_file(os.path.join(args.hf_dir, "config.json")),
      "endpoint": args.endpoint,
      "served_model": args.served_model,
      "k": K,
      "gate": {"top1_min": TOP1_MIN, "kl_max_exclusive": KL_MAX},
      "passed": passed,
      "summary": summary,
      "per_prompt": rows,
      "jax": {"version": jax.__version__, "devices": len(jax.devices()), "processes": jax.process_count(),
              "mesh": dict(mesh.shape), "load_seconds": load_s},
      "kl_note": "KL over vLLM top-K ids + rest bin; a lower bound on full-vocab KL.",
      "written_at": time.time(),
  }
  pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
  pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
  if args.dump_npz:
    raw = {}
    for i, ids in enumerate(all_ids):
      top_ids, top_lp, tok_lp = vl[i]
      jlp, jam = per_prompt[i]
      L = len(ids)
      raw |= {f"p{i}_ids": np.asarray(ids, np.int32), f"p{i}_vllm_top_ids": top_ids, f"p{i}_vllm_top_lp": top_lp,
              f"p{i}_vllm_tok_lp": tok_lp, f"p{i}_jax_lp": np.asarray(jlp[: L - 1]), f"p{i}_jax_argmax": np.asarray(jam[: L - 1])}
    np.savez_compressed(args.dump_npz, **raw)
  print(json.dumps({"passed": passed, **summary}), flush=True)
  return 0 if passed else 1


if __name__ == "__main__":
  sys.exit(main())
