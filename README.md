# Rethinking Self-Repair in Code-as-Policy Agents: What Drives Retry Success?

<p align="center">
  <a href="https://arxiv.org/abs/2611.06863"><img src="https://img.shields.io/badge/arxiv-2611.06863-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv"></a>
  &nbsp;
  <a href="https://learn-at-deploy.github.io/"><img src="https://img.shields.io/badge/Learn%40Deploy%20%40%20CoRL%202026-2e7d32?style=flat" alt="Learn@Deploy @ CoRL 2026"></a>
</p>

RoboRepair studies how robot code agents recover from failed programs. **In our experiments, simple task guidance and feedback tailored to the failure yield similar recovery rates.**

<p align="center">
  <img src="assets/retry_overview.png" width="100%" alt="The retry process and success rates for each approach." />
</p>

*How retries work and how often each approach succeeds.*

---

## Install

```bash
git clone https://github.com/siddharthaphale/roborepair.git && cd roborepair
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r envs/requirements-gpu.txt -e .
```

## Running experiments

Run the neutral-prompt comparison:

```bash
MUJOCO_GL=disable python experiments/feedback_comparison.py \
  --config configs/neutral_prompt_study/neutral_prompt_comparison.yaml \
  --seed 0 \
  --output-dir runs/neutral_prompt_comparison_seed0
```

See [experiments/](experiments/) for the other studies and [configs/](configs/) for their settings.

## Citation and license

Please cite our [paper](https://arxiv.org/abs/2611.06863) when using this work. Code is released under the [MIT license](LICENSE).
