"""trainer/train_tpu_vm.sh commands and the trainer/train.py merge entry point (model loading stubbed)."""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import types

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
TRAINER = REPO / "trainer"
sys.path.insert(0, str(TRAINER))

IMAGE = ("asia-northeast1-docker.pkg.dev/northam-ce-mlai-tpu/jawadamin-dsv4/"
         "trainer-jawadamin-us-south1-0926:8da2b99b5514")
DOCKER_PREFIX = ["run", "--rm", "--name", "kd-train", "--privileged", "--net=host", "--ipc=host", "-v",
                 "/mnt/data:/mnt/data", "-w", "/mnt/data/code", "-e", "GCS_CACHE=/mnt/data/gcs", "-e",
                 "EXPORT_STREAM=1", "-e", "JAX_COMPILATION_CACHE_DIR=/mnt/data/jaxcache-train", IMAGE, "bash", "-lc"]


def _launch(tmp_path, *argv, eval_tasks=True, **env_extra):
  """Run train_tpu_vm.sh with stub docker / mkdir on PATH; return (process, `docker run` argv or None)."""
  stub = tmp_path / "bin"
  stub.mkdir(exist_ok=True)
  (stub / "docker").write_text('#!/usr/bin/env bash\nif [ "$1" = run ]; then printf "%s\\0" "$@" > "$DOCKER_LOG"; fi\n'
                               'echo "$DOCKER_HOST" > "$DOCKER_LOG.host"\n')
  (stub / "mkdir").write_text("#!/usr/bin/env bash\nexit 0\n")  # the launcher creates /mnt/data dirs
  for f in stub.iterdir():
    f.chmod(0o755)
  (tmp_path / "eval.json").write_text("[]")
  log = tmp_path / "docker.log"
  if log.exists():
    log.unlink()
  env = {"PATH": f"{stub}:/usr/bin:/bin", "HOME": str(tmp_path), "DOCKER_LOG": str(log), **env_extra}
  if eval_tasks:
    env.setdefault("EVAL_TASKS", str(tmp_path / "eval.json"))
  proc = subprocess.run(["bash", str(TRAINER / "train_tpu_vm.sh"), *argv], env=env, capture_output=True, text=True)
  return proc, (log.read_text().split("\0")[:-1] if log.exists() else None)


def _arg(cmd, flag):
  return cmd[cmd.index(flag) + 1]


def test_sft_command_and_defaults(tmp_path):
  proc, argv = _launch(tmp_path, "sft", "r1", "/d.jsonl", "--warmup_frac", "0.05")
  assert proc.returncode == 0, proc.stderr
  assert argv[:-1] == DOCKER_PREFIX
  assert (tmp_path / "docker.log.host").read_text().strip() == "unix:///var/run/docker.sock"
  ev = str(tmp_path / "eval.json")
  assert argv[-1] == (
      "set -o pipefail; python3 trainer/train.py sft --hf_dir /mnt/data/models/Qwen3-1.7B --out_dir /mnt/data/train/r1 "
      "--run_id r1 --fsdp 8 --tp 1 --rank 256 --alpha 512 --remat DECODER --data /d.jsonl       --micro_batch 8 "
      f"--max_seq_len 8192 --eval_tasks {ev} --warmup_frac 0.05 --export_merged --gcs_out "
      "gs://ckpt-jawadamin-us-east5-0926/Qwen3-1.7B/r1 2>&1 | tee -a /mnt/data/train/r1/train.log")


def test_sft_overrides(tmp_path):
  proc, argv = _launch(tmp_path, "sft", "r2", "/d.jsonl", MODEL="Qwen3-4B-Instruct-2507", FSDP="4", TP="2",
                       MICRO_BATCH="4", MAX_SEQ_LEN="2048", GCS_ROOT="gs://x/y", RANK="512", REMAT="BLOCK",
                       CODE="/mnt/data/other")
  assert proc.returncode == 0, proc.stderr
  assert argv[argv.index("-w") + 1] == "/mnt/data/other"
  cmd = argv[-1].split()
  assert _arg(cmd, "--hf_dir") == "/mnt/data/models/Qwen3-4B-Instruct-2507"
  assert (_arg(cmd, "--fsdp"), _arg(cmd, "--tp"), _arg(cmd, "--micro_batch"), _arg(cmd, "--max_seq_len")) == (
      "4", "2", "4", "2048")
  assert (_arg(cmd, "--rank"), _arg(cmd, "--alpha"), _arg(cmd, "--remat")) == ("512", "1024", "BLOCK")
  assert _arg(cmd, "--gcs_out") == "gs://x/y/r2"
  proc, argv = _launch(tmp_path, "sft", "r3", "/d.jsonl", BASE="/m/b")
  assert _arg(argv[-1].split(), "--hf_dir") == "/m/b"


def test_eval_tasks_is_required_for_sft_and_grpo(tmp_path):
  for argv in (("sft", "r1", "/d.jsonl"), ("grpo", "r1", "/p", "/r.jsonl", "0")):
    proc, docker = _launch(tmp_path, *argv, eval_tasks=False, REF="/m/kd")
    assert proc.returncode == 2 and "EVAL_TASKS is required" in proc.stderr and docker is None
    proc, docker = _launch(tmp_path, *argv, EVAL_TASKS=str(tmp_path / "missing.json"), REF="/m/kd")
    assert proc.returncode == 2 and "missing" in proc.stderr and docker is None


def test_grpo_command(tmp_path):
  proc, argv = _launch(tmp_path, "grpo", "r1", "/p", "/r.jsonl", "2", "--beta", "0.02", REF="/mnt/data/models/kd")
  assert proc.returncode == 0, proc.stderr
  assert argv[:-1] == DOCKER_PREFIX
  ev = str(tmp_path / "eval.json")
  assert argv[-1] == (
      "set -o pipefail; python3 trainer/train.py grpo_iter --hf_dir /p --out_dir /mnt/data/train/r1 --run_id r1 "
      "--fsdp 8 --tp 1 --rank 256 --alpha 512 --remat DECODER --micro_batch 8 --max_seq_len 8192       "
      f"--rollouts /r.jsonl --ref_hf_dir /mnt/data/models/kd --iteration 2 --eval_tasks {ev} --beta 0.02       "
      "--gcs_out gs://ckpt-jawadamin-us-east5-0926/Qwen3-1.7B/r1/grpo/iter_3 2>&1 | tee -a "
      "/mnt/data/train/r1/train.log")


def test_grpo_needs_a_reference_other_than_the_base(tmp_path):
  proc, argv = _launch(tmp_path, "grpo", "r1", "/p", "/r.jsonl", "0")
  assert proc.returncode == 2 and "REF" in proc.stderr and argv is None
  for ref in ("/mnt/data/models/Qwen3-1.7B", "/mnt/data/models/Qwen3-1.7B/"):
    proc, argv = _launch(tmp_path, "grpo", "r1", "/p", "/r.jsonl", "0", REF=ref)
    assert proc.returncode == 2 and "ALLOW_BASE_REF" in proc.stderr and argv is None
  proc, argv = _launch(tmp_path, "grpo", "r1", "/p", "/r.jsonl", "0", REF="/m/b", BASE="/m/b")
  assert proc.returncode == 2 and argv is None
  proc, argv = _launch(tmp_path, "grpo", "r1", "/p", "/r.jsonl", "0", REF="/m/b", BASE="/m/b", ALLOW_BASE_REF="1")
  assert proc.returncode == 0, proc.stderr
  assert _arg(argv[-1].split(), "--ref_hf_dir") == "/m/b"


def test_merge_and_probe_commands(tmp_path):
  proc, argv = _launch(tmp_path, "merge", "r1", "/pi_2", "gs://b/r1/grpo/iter_3/ckpt", "iter_3", "--z",
                       eval_tasks=False)
  assert proc.returncode == 0, proc.stderr
  assert argv[:-1] == DOCKER_PREFIX  # EXPORT_STREAM=1 like the other phases
  assert argv[-1] == ("set -o pipefail; python3 trainer/train.py merge --hf_dir /pi_2 --out_dir /mnt/data/train/r1 "
                      "--run_id r1 --fsdp 8 --tp 1 --rank 256 --alpha 512 --remat DECODER --ckpt "
                      "gs://b/r1/grpo/iter_3/ckpt --tag iter_3 --z       --gcs_out "
                      "gs://ckpt-jawadamin-us-east5-0926/Qwen3-1.7B/r1/grpo/iter_3 2>&1 | tee -a "
                      "/mnt/data/train/r1/train.log")
  proc, argv = _launch(tmp_path, "probe", "r1", "--seq_lens", "8192", eval_tasks=False)
  assert proc.returncode == 0, proc.stderr
  assert argv[-1] == ("set -o pipefail; python3 trainer/train.py probe_memory --hf_dir /mnt/data/models/Qwen3-1.7B "
                      "--out_dir /mnt/data/train/r1 --run_id r1 --fsdp 8 --tp 1 --rank 256 --alpha 512 --remat DECODER "
                      "--gcs_out gs://ckpt-jawadamin-us-east5-0926/Qwen3-1.7B/r1 --seq_lens 8192 2>&1 | tee -a "
                      "/mnt/data/train/r1/train.log")


def test_refuses_unaligned_rank_and_unknown_phase(tmp_path):
  proc, argv = _launch(tmp_path, "sft", "r1", "/d.jsonl", RANK="128")
  assert proc.returncode == 2 and "multiple of 256" in proc.stderr and argv is None
  proc, argv = _launch(tmp_path, "train", "r1")
  assert proc.returncode == 2 and "phase must be" in proc.stderr and argv is None


# ---------------------------------------------------------------------------
# train.py merge entry point
# ---------------------------------------------------------------------------


def _run_merge(tmp_path, monkeypatch, extra_env, extra_args):
  """Run train.py's `merge` subcommand with model loading, LoRA and export stubbed; return the calls."""
  pytest.importorskip("jax")
  transformers = pytest.importorskip("transformers")
  import student_models as sm  # pylint: disable=g-import-not-at-top
  import train  # pylint: disable=g-import-not-at-top

  rec = {}
  fake_tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=0)
  monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda d: fake_tok)
  monkeypatch.setattr(sm, "maybe_init_distributed", lambda: None)
  monkeypatch.setattr(sm, "make_mesh", lambda fsdp, tp: types.SimpleNamespace(shape={"fsdp": fsdp, "tp": tp}))
  monkeypatch.setattr(sm, "load_qwen3", lambda *a, **k: rec.setdefault("loader", (a[0], k)) and "model")
  monkeypatch.setattr(sm, "apply_lora", lambda model, mesh, tgt, *a, **k: rec.setdefault("targets", tgt) and ("lora", model))
  monkeypatch.setattr(sm, "lora_deltas_hf", lambda *a, **k: rec.setdefault("deltas", a[1:]) and {})
  monkeypatch.setattr(train.tc, "resolve_inputs", lambda args: None)
  monkeypatch.setattr(train.tc, "finalize_outputs", lambda args: rec.setdefault("finalized", args.gcs_out))
  monkeypatch.setattr(train.tc, "restore_lora", lambda lora, root: rec.setdefault("restored", (lora, root)) and 5)

  def export(hf_dir, out_dir, fn, tag, stream_to=None):
    fn()
    rec["export"] = (hf_dir, out_dir, tag, stream_to)

  monkeypatch.setattr(train.tc, "export_merged", export)
  monkeypatch.delenv("EXPORT_STREAM", raising=False)
  for k, v in extra_env.items():
    monkeypatch.setenv(k, v)
  monkeypatch.setattr(sys, "argv", ["train.py", "merge", "--hf_dir", "/m/pi_2", "--out_dir", str(tmp_path / "o"),
                                    "--run_id", "r", "--ckpt", "gs://b/run/grpo/iter_3/ckpt", "--tag", "iter_3",
                                    "--fsdp", "8", "--tp", "1", "--rank", "256", "--alpha", "512", *extra_args])
  assert train.main() == 0
  return rec


def test_merge_restores_the_checkpoint_and_streams_the_export(tmp_path, monkeypatch):
  import student_models as sm  # pylint: disable=g-import-not-at-top

  rec = _run_merge(tmp_path, monkeypatch, {"EXPORT_STREAM": "1"}, ["--gcs_out", "gs://b/run/grpo/iter_3"])
  assert rec["loader"][0] == "/m/pi_2" and rec["targets"] == sm.LORA_TARGETS
  assert rec["restored"] == (("lora", "model"), "gs://b/run/grpo/iter_3/ckpt")
  assert rec["deltas"] == (512.0, 256)
  assert rec["export"] == ("/m/pi_2", str(tmp_path / "o"), "iter_3", "gs://b/run/grpo/iter_3/merged-iter_3")
  assert rec["finalized"] == "gs://b/run/grpo/iter_3"


def test_merge_without_export_stream_writes_locally(tmp_path, monkeypatch):
  rec = _run_merge(tmp_path, monkeypatch, {}, ["--gcs_out", "gs://b/run/grpo/iter_3"])
  assert rec["export"][3] is None
  assert not os.environ.get("EXPORT_STREAM")
