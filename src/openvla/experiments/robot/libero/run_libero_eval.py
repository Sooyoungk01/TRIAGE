import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image  # noqa: E402
from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action, set_seed_everywhere  # noqa: E402
from experiments.robot.triage_utils import TriageOpenVLA  # noqa: E402

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}
OVERRIDES = ("b_vis", "n_important", "first_bvis", "prune_layer", "max_age", "delta", "maint_ratio", "hysteresis")


def act_hash(a):
    return hashlib.sha1(np.asarray(a, dtype=np.float32).tobytes()).hexdigest()[:12]


def run_episode(policy, env, init_state, task, args, unnorm_key, log):
    env.reset()
    obs = env.set_init_state(init_state)
    policy.reset()
    t = 0
    while t < MAX_STEPS[args.task_suite_name] + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue
        action = policy(get_libero_image(obs, 224), task, unnorm_key, crop=True)
        log(step=t, act_hash=act_hash(action), **policy.info)
        action = invert_gripper_action(normalize_gripper_action(action, binarize=True))
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True
        t += 1
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_suite_name", choices=tuple(MAX_STEPS), default="libero_spatial")
    ap.add_argument("--pretrained_checkpoint", default=None, help="Hub id or local dir")
    ap.add_argument("--task_ids", default=None, help="comma-separated, default all")
    ap.add_argument("--trial_start", type=int, default=0)
    ap.add_argument("--num_trials_per_task", type=int, default=50)
    ap.add_argument("--keyframe", action="store_true", help="TRIAGE+KF")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log_steps", action="store_true")
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    for k in OVERRIDES:
        ap.add_argument(f"--{k}", type=float, default=None)
    args = ap.parse_args()

    set_seed_everywhere(args.seed)
    suite_name = args.task_suite_name
    params = {k: (float(v) if k in ("delta", "maint_ratio", "hysteresis") else int(v))
              for k in OVERRIDES if (v := getattr(args, k)) is not None}
    params["keyframe"] = args.keyframe
    ckpt = args.pretrained_checkpoint or f"openvla/openvla-7b-finetuned-{suite_name.replace('_', '-')}"
    policy = TriageOpenVLA(ckpt, params)
    unnorm_key = policy.unnorm_key(suite_name)

    last = args.trial_start + args.num_trials_per_task - 1
    out = args.out or str(ROOT / "results" / f"{suite_name}{'_kf' if args.keyframe else ''}_trials{args.trial_start}-{last}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done_eps = set()
    if os.path.isfile(out):
        done_eps = {(r["task_id"], r["episode"]) for r in map(json.loads, open(out)) if r["type"] == "episode"}
    f = open(out, "a")

    def write(**rec):
        f.write(json.dumps(rec) + "\n")
        f.flush()

    suite = benchmark.get_benchmark_dict()[suite_name]()
    task_ids = range(suite.n_tasks) if args.task_ids is None else [int(x) for x in args.task_ids.split(",")]
    n = succ = 0
    for task_id in task_ids:
        init_states = suite.get_task_init_states(task_id)
        env, task = get_libero_env(suite.get_task(task_id), "openvla", resolution=256)
        for ep in range(args.trial_start, args.trial_start + args.num_trials_per_task):
            if (task_id, ep) in done_eps:
                continue

            def log(**rec):
                if args.log_steps:
                    write(type="step", task_id=task_id, episode=ep, **rec)

            try:
                success = run_episode(policy, env, init_states[ep], task, args, unnorm_key, log)
            except Exception as e:
                print(f"episode error: {e!r}")
                success = False
            write(type="episode", task_id=task_id, episode=ep, success=bool(success))
            n += 1
            succ += int(success)
            print(f"task {task_id} episode {ep}: {'success' if success else 'failure'} | {succ}/{n} ({100 * succ / n:.1f}%)")


if __name__ == "__main__":
    main()
