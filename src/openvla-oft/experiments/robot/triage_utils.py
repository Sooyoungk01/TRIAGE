import contextlib
import os
from types import SimpleNamespace

from huggingface_hub import snapshot_download

from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector
from experiments.robot.robot_utils import get_action, get_model
from prismatic.extern import triage
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRIAGE_PARAMS = dict(prune_layer=3, b_vis=163, n_important=233, first_bvis=163, max_age=10, delta=0.3, maint_ratio=0.06,
                     hysteresis=1.25, lam=1.0)
TRIAGE_PARAMS_GOAL = dict(b_vis=91, n_important=161)


def local_checkpoint(name):
    if os.path.isdir(name):
        return os.path.abspath(name)
    path = os.path.join(ROOT, "checkpoints", name.split("/")[-1])
    if not os.path.isfile(os.path.join(path, "config.json")):
        snapshot_download(repo_id=name, local_dir=path)
    return path


@contextlib.contextmanager
def in_root():
    old = os.getcwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(old)


class TriageOFT:
    def __init__(self, checkpoint, suite, params=None):
        triage.install()
        self.cfg = SimpleNamespace(
            model_family="openvla", pretrained_checkpoint=local_checkpoint(checkpoint), use_l1_regression=True,
            use_diffusion=False, num_diffusion_steps_train=50, num_diffusion_steps_inference=50, use_film=False,
            num_images_in_input=2, use_proprio=True, center_crop=True, num_open_loop_steps=8, lora_rank=32,
            unnorm_key=suite, load_in_8bit=False, load_in_4bit=False)
        with in_root():
            self.vla = get_model(self.cfg)
        self.proprio_projector = get_proprio_projector(self.cfg, self.vla.llm_dim, proprio_dim=8)
        self.action_head = get_action_head(self.cfg, self.vla.llm_dim)
        self.processor = get_processor(self.cfg)
        key = suite if suite in self.vla.norm_stats else f"{suite}_no_noops"
        assert key in self.vla.norm_stats, f"no un-normalization key for {suite}"
        self.cfg.unnorm_key = key
        vb = self.vla.vision_backbone
        n_vis = vb.get_num_patches() * vb.get_num_images_in_input()
        base = dict(TRIAGE_PARAMS, **(TRIAGE_PARAMS_GOAL if suite == "libero_goal" else {}))
        self.params = dict(base, **(params or {}), vis_start=1, vis_end=1 + n_vis, n_action=NUM_ACTIONS_CHUNK * ACTION_DIM)
        self.info = None
        self.reset()

    def reset(self):
        self.prev = None

    def __call__(self, observation, task_label):
        st = dict(self.params, prev=self.prev)
        self.vla.language_model.config.triage = st
        actions = get_action(self.cfg, self.vla, observation, task_label, processor=self.processor,
                             action_head=self.action_head, proprio_projector=self.proprio_projector,
                             noisy_action_projector=None, use_film=False)
        self.vla.language_model.config.triage = None
        self.prev, self.info = st["out"], st["info"]
        return actions
