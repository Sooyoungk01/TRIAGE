# TRIAGE

TRIAGE is a training-free inference method for vision-language-action (VLA) models. At each policy query it decides, for every visual token, whether to recompute it, reuse its cached keys and values from an earlier query, or prune it.

At each query, the first `K = 3` layers run densely. TRIAGE then ranks the visual tokens by importance: the layer-`K-1` attention of the action query, plus a deep-layer forecast carried over from the episode's first query. The top `N` tokens form the important set. A hysteresis factor keeps the set stable across queries.

`B` visual tokens are recomputed in layers `K..L-1`. They are chosen in this order:

1. A maintenance share (`r/B = 0.06` of the budget) refreshes the oldest cached tokens outside the important set.
2. Important tokens without a valid cache. A cache is invalid if the token is new to the set, was computed more than `T = 10` queries ago, or has a layer-`K-1` value drift of at least `δ = 0.3`.
3. The remaining important tokens, ordered by how much they changed.

Every other visual token is reused as a key if its cache is valid, and pruned otherwise.

## Layout

```
src/openvla/        OpenVLA, with TRIAGE added
src/openvla-oft/    OpenVLA-OFT, with TRIAGE added
simpler/            patch for SimplerEnv-OpenVLA
scripts/            reproduce_libero.sh
```

The TRIAGE files inside each tree are:

| File | Purpose |
| --- | --- |
| `prismatic/extern/triage.py` | TRIAGE engine. It replaces `LlamaModel.forward` at runtime. |
| `experiments/robot/triage_utils.py` | Model loading and per-query state. |
| `experiments/robot/libero/run_libero_eval.py` | LIBERO evaluation with TRIAGE. |

In `src/openvla`, `prismatic/extern/hf/modeling_prismatic.py` is the VLA-Cache version, which passes the cache through `generate`. Everything else is the upstream code.

## Installation

Each model needs its own environment, because the two trees ship different `prismatic` packages.

```bash
# OpenVLA
conda create -n triage-openvla python=3.10 -y && conda activate triage-openvla
pip install -e src/openvla

# OpenVLA-OFT
conda create -n triage-oft python=3.10 -y && conda activate triage-oft
pip install -e src/openvla-oft
```

Then install LIBERO in each environment:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git && pip install -e LIBERO
pip install -r src/openvla/experiments/robot/libero/libero_requirements.txt
```

Both `pyproject.toml` files pin:

- PyTorch 2.2.0
- `tokenizers` 0.21.1
- the VLA-Cache `transformers` fork at commit `2302fce`

## LIBERO

Run from `src/openvla` or `src/openvla-oft`. Checkpoints are downloaded into `checkpoints/` on first use:

- OpenVLA: `openvla/openvla-7b-finetuned-libero-*`
- OpenVLA-OFT: `moojink/openvla-7b-oft-finetuned-libero-*`

```bash
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

cd src/openvla
python experiments/robot/libero/run_libero_eval.py --task_suite_name libero_spatial

cd src/openvla-oft
python experiments/robot/libero/run_libero_eval.py --task_suite_name libero_goal
python experiments/robot/libero/run_libero_eval.py --task_suite_name libero_10 --keyframe   # TRIAGE+KF
```

| Option | Meaning |
| --- | --- |
| `--task_ids 0,3` | Run only these tasks. |
| `--trial_start` / `--num_trials_per_task` | Choose which initial states to run. The default is all 50. |
| `--pretrained_checkpoint` | Use a local directory or another Hub id. |
| `--log_steps` | Also write one record per query: action hash, recomputed rows, prefill time in ms. |
| `--keyframe` | Run a dense query whenever the remapped forecast and the current layer-0 saliency disagree (cosine below 0.92). |

Results go to `results/<suite>[_kf]_trials<a>-<b>.jsonl`, with one `episode` record per episode. A run that is interrupted resumes from the episodes already logged.

`scripts/reproduce_libero.sh openvla` and `scripts/reproduce_libero.sh openvla-oft` run all four suites. Each suite is split into trials 0-24 and 25-49, as in the paper runs. Add `--keyframe` for TRIAGE+KF.

### Hyperparameters

| | OpenVLA | OFT | OFT, LIBERO-Goal |
| --- | --- | --- | --- |
| dense layers `K` | 3 | 3 | 3 |
| recomputed visual tokens `B` | 88 | 163 | 91 |
| important set `N` | 158 | 233 | 161 |
| first-query tokens | 227 | 163 | 163 |
| cache age bound `T` | 10 | 10 | 10 |
| drift gate `δ` | 0.3 | 0.3 | 0.3 |
| maintenance share `r/B` | 0.06 | 0.06 | 0.06 |
| hysteresis | 1.25 | 1.25 | 1.25 |

These are the defaults in `triage_utils.py`. Each one can be overridden from the command line, for example `--b_vis 64 --n_important 134`.

The paper numbers were produced with OSMesa rendering. EGL rendering can change pixels slightly, and with them the rollouts.

## SimplerEnv

The SimplerEnv runs use the base `openvla/openvla-7b` checkpoint with the OpenVLA settings above.

1. Set up [SimplerEnv-OpenVLA](https://github.com/DelinQu/SimplerEnv-OpenVLA) following its README, in an environment where `src/openvla` is installed.
2. Put `src/openvla` on `PYTHONPATH`.
3. Apply the patch and run the evaluations:

```bash
cd SimplerEnv-OpenVLA
git checkout ccfe380
git apply /path/to/TRIAGE/simpler/simpler_openvla.patch
export PYTHONPATH=/path/to/TRIAGE/src/openvla:$PYTHONPATH

bash scripts/move_near_visual_matching.sh openvla/openvla-7b openvla -0.8 <log_dir> <gpu>
bash scripts/move_near_variant_agg_240.sh openvla/openvla-7b openvla -0.8 <log_dir> <gpu>
bash scripts/pick_coke_can_variant_agg_240.sh openvla/openvla-7b openvla -0.8 <log_dir> <gpu>
```

The patch does three things:

- Replaces the policy call in `openvla_model.py` with TRIAGE, which uses OpenVLA's standard prompt.
- Resets the cache at the start of every episode.
- Adds the two Variant Aggregation scripts with 240 episodes.

## Acknowledgements

This code builds on:

- [OpenVLA](https://github.com/openvla/openvla)
- [OpenVLA-OFT](https://github.com/moojink/openvla-oft)
- [VLA-Cache](https://github.com/siyuhsu/vla-cache), whose `transformers` fork and OpenVLA model file are used here
- [SAFE-Pruner](https://github.com/msssl/SAFE-Pruner), whose deep-layer forecast TRIAGE uses for importance
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [SimplerEnv](https://github.com/simpler-env/SimplerEnv)

The code under `src/openvla` and `src/openvla-oft` keeps its original MIT license (see the `LICENSE` file in each directory).
