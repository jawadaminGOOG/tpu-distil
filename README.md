# TPU-Distil: Agentic On-Policy Distillation for TPU-Optimized Models

`TPU-Distil` is an end-to-end framework for distilling frontier GPU-hosted teacher models (`DeepSeek-V3`, `GLM-5`, `Kimi-K3`) into TPU-optimized student models (`Qwen3-Next-80B-A3B` on **MaxText** and **Tunix**) using **Student-Centered One-step Reinforcement (`SCoRe-RL`)**.

## Core Capabilities

- **Hardware-Aligned Student Architecture (`Qwen3-Next-80B-A3B`)**: Leverages hybrid linear attention (Gated DeltaNet) and ultra-sparse Mixture-of-Experts (80B total parameters, 3B active per token) with dimensions aligned to the 256-dimension MXU systolic array on TPU `v6e` and `v7x`.
- **Cross-Accelerator SCoRe Pipeline**:
  1. **Cold-Start Behavior Cloning (`BC-Init`)**: Bootstraps `[Thought, Action, Observation]` tool-use syntax from teacher demonstrations with observation-masked SFT loss.
  2. **Mentored Problem-Solving (`MPS`)**: Generates high-throughput on-policy student rollouts on TPU (`vLLM-TPU` / `JetStream`) while querying the GPU teacher only to localize and replace the **earliest critical error step** ($\sigma_k \to \sigma'_k$).
  3. **Short-Horizon Key-Step GRPO (`SCoRe-RL`)**: Runs JAX-native GRPO in **Tunix** starting from the verified prefix $(\sigma_1, \dots, \sigma_{k-1})$ with composite rewards ($R_{\text{final}} + R_{\text{key}} + R_{\text{format}}$), reducing policy gradient variance and bounding compounding errors from $O(H^2)$ to $O(H)$.
- **Multi-Domain Agent Environments**: Unified `CodeAct` execution harness supporting **FinanceBench** (local Python + retrieval sandbox) and **Terminal-Bench / SWE-bench** (containerized bash + pytest sandboxes).

## Repository Layout

```text
TPU-Distil/
├── docs/
│   └── ARCHITECTURE.md       # Evergreen system architecture and technical design
├── src/tpu_distil/
│   ├── schema/               # Trajectory, step, and MPS intervention schemas
│   ├── environments/         # CodeAct sandboxes (FinanceBench, Terminal-Bench, SWE-bench)
│   ├── mps/                  # Student rollout + teacher first-error splicing coordinator
│   ├── sft/                  # MaxText / Tunix observation-masked SFT adapters
│   └── rl/                   # Tunix short-horizon GRPO & key-step reward functions
├── tests/                    # Unit and integration verification suites
├── AGENTS.md                 # Codebase invariants and engineering guardrails
└── REVIEW.md                 # Pre-merge verification passes
```

## Documentation

See [docs/ARCHITECTURE.md](file:///usr/local/google/home/jawadamin/Repos/TPU-Distil/docs/ARCHITECTURE.md) for the complete system design, hardware alignment rationale, and mathematical formulation.
