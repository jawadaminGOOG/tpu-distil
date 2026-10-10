"""compute_report.py: FLOP model, accelerator time, teacher up-time and cost on a tiny hand-checked manifest."""
import json
import subprocess
import sys
from pathlib import Path

DISTIL = Path(__file__).resolve().parents[1] / "distil"
sys.path.insert(0, str(DISTIL))
import compute_report as cr  # noqa: E402

# h=4, L=1, ffn=8, vocab=10, 2 heads x head_dim 2 (q_out 4), 1 kv head (kv_out 2), rank 1.
CFG = {"hidden_size": 4, "num_hidden_layers": 1, "intermediate_size": 8, "vocab_size": 10, "num_attention_heads": 2,
       "head_dim": 2, "num_key_value_heads": 1, "tie_word_embeddings": True}


def _sku(sku, usd):
  return {"skuId": sku, "description": sku, "pricingInfo": [{"effectiveTime": "t", "pricingExpression": {
      "tieredRates": [{"unitPrice": {"units": str(int(usd)), "nanos": round((usd - int(usd)) * 1e9)}}]}}]}


def test_dims_and_flops():
  d = cr.model_dims(CFG, 1)
  assert d["decoder_matmul_params"] == 144 and d["lm_head_params"] == 40 and d["lora_params"] == 64
  assert cr.forward_flops(d, 10) == 2 * 184 + 4 * 1 * 4 * 10 / 2  # 448
  assert cr.update_flops(d, 10) == (2 * 448 + 3 * 128, 2 * 448 + 3 * 128 + 448 + 128)


def test_manifest_end_to_end(tmp_path):
  w = lambda name, obj: (tmp_path / name).write_text(json.dumps(obj)) or name  # noqa: E731
  w("config.json", CFG)
  (tmp_path / "prices").mkdir()
  (tmp_path / "prices" / "p.json").write_text(json.dumps({"skus": [_sku("TPU", 2.0), _sku("GPU", 4.0), _sku("GPU2", 6.0)]}))
  w("sft_stats.json", {"tokens_real": 100, "tokens_trained": 20, "n_kept": 10, "epochs": 2})
  w("sft_done.json", {"wall_s": 3600})
  w("grpo_stats.json", {"tokens_real": 50, "tokens_trained": 10, "n_sequences": 5, "policy_hf_dir": "a", "ref_hf_dir": "b"})
  w("grpo_done.json", {"wall_s": 1800})
  (tmp_path / "gen.jsonl").write_text("\n".join(json.dumps(r) for r in [
      {"id": "1", "choices": [{"content": "x"}, {"content": "y"}], "usage": {"prompt_tokens": 6, "completion_tokens": 4}},
      {"id": "2", "error": "HTTP 500"}]))
  (tmp_path / "gen.log").write_text("prompts=2\n100/2 ok=1 err=0 elapsed=360s out_tok/s=1\n")
  (tmp_path / "teacher.jsonl").write_text(json.dumps({"id": "1", "choices": [{"content": "z"}],
                                                      "usage": {"prompt_tokens": 3, "completion_tokens": 7}}))
  (tmp_path / "teacher.log").write_text("1/1 ok=1 err=0 elapsed=900s out_tok/s=1\n")
  (tmp_path / "kw.log").write_text("\n".join([
      "2026-10-10T00:00:00Z 429 starting", "2026-10-10T00:05:00Z 200 ok", "2026-10-10T00:10:00Z 200 ok",
      "2026-10-10T00:15:00Z 200 ok",                     # window 1: 10 min span + 5 = 15 min
      "2026-10-10T02:00:00Z 200 ok",                     # gap > 2 intervals: window 2: 0 + 5 = 5 min
      "2026-10-10T02:05:00Z 429 down"]))
  man = {"model_config": "config.json", "lora_rank": 1, "chip": {"name": "c", "peak_bf16_tflops": 1e-9, "source": "s"},
         "prices_dir": "prices", "tpu_sku": "TPU", "gpu_sku": "GPU", "gpu_sku_alt": "GPU2", "teacher_gpus": 8,
         "sft": [{"name": "kd", "role": "recipe", "stats": "sft_stats.json", "done": "sft_done.json", "chips": 8}],
         "grpo": [{"name": "rl", "role": "recipe", "stats": "grpo_stats.json", "done": "grpo_done.json", "chips": 8}],
         "generation": [{"name": "roll", "role": "recipe", "device": "tpu", "chips": 4, "gen": "gen.jsonl", "log": "gen.log"},
                        {"name": "teach", "role": "recipe", "device": "teacher", "chips": 8, "gen": "teacher.jsonl",
                         "log": "teacher.log"}],
         "teacher_uptime_logs": ["kw.log"], "data": []}
  w("manifest.json", man)
  subprocess.run([sys.executable, str(DISTIL / "compute_report.py"), "--manifest", str(tmp_path / "manifest.json"),
                  "--out", str(tmp_path / "o.json"), "--md", str(tmp_path / "o.md")], check=True, capture_output=True)
  o = json.loads((tmp_path / "o.json").read_text())
  runs = {r["name"]: r for r in o["runs"]}
  assert runs["kd"]["tokens_processed"] == 200 and runs["kd"]["model_flops"] == 200 * 1280
  assert runs["kd"]["hardware_flops"] == 200 * 1856 and runs["kd"]["accel_hours"] == 8.0
  assert runs["rl"]["model_flops"] == 50 * 1280 + 2 * 50 * 448  # update + old-policy and reference passes
  assert runs["roll"]["accel_hours"] == 360 * 4 / 3600 and runs["roll"]["requests_samples"] == 2
  assert runs["roll"]["model_flops"] == 10 * cr.forward_flops(cr.model_dims(CFG, 1), 5)
  assert runs["teach"]["accel_hours"] == 2.0 and "model_flops" not in runs["teach"]
  r = o["recipe"]
  assert r["tpu_chip_hours"] == 8.0 + 4.0 + 0.4 and r["teacher_gpu_hours_active"] == 2.0
  assert r["cost_usd_tpu"] == 12.4 * 2.0 and r["cost_usd_teacher_active"] == 8.0 and r["cost_usd_teacher_active_alt"] == 12.0
  te = o["teacher_endpoint"]
  assert abs(te["uptime_hours"] - (15 + 5) / 60) < 1e-9 and len(te["windows"]) == 2
  assert abs(te["gpu_hours"] - 8 * 20 / 60) < 1e-9
  assert abs(runs["kd"]["mfu_model"] - 200 * 1280 / (8 * 3600 * 1e3)) < 1e-12
