import json
import os

import numpy as np
import tensorflow as tf
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor, DynamicCache

from experiments.robot.openvla_utils import crop_and_resize
from prismatic.extern import triage
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

DEVICE = torch.device("cuda:0")
NUM_PATCHES = 256
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRIAGE_PARAMS = dict(prune_layer=3, b_vis=88, n_important=158, first_bvis=227, max_age=10, delta=0.3, maint_ratio=0.06,
                     hysteresis=1.25, lam=1.0, keyframe=False, kf_threshold=0.92)


def local_checkpoint(name):
    if os.path.isdir(name):
        return os.path.abspath(name)
    path = os.path.join(ROOT, "checkpoints", name.split("/")[-1])
    if not os.path.isfile(os.path.join(path, "config.json")):
        snapshot_download(repo_id=name, local_dir=path)
    return path


def center_crop(image, scale=0.9):
    image = tf.convert_to_tensor(np.array(image))
    dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, scale, 1)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, dtype, saturate=True)
    return Image.fromarray(image.numpy()).convert("RGB")


class TriageOpenVLA:
    def __init__(self, checkpoint, params=None):
        triage.install()
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        checkpoint = local_checkpoint(checkpoint)
        self.vla = OpenVLAForActionPrediction.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(DEVICE)
        stats = os.path.join(checkpoint, "dataset_statistics.json")
        if os.path.isfile(stats):
            with open(stats) as f:
                self.vla.norm_stats = json.load(f)
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        cfg = self.vla.language_model.config
        cfg.proportion_attn_var = None
        cfg.reusable_patches = None
        self.params = dict(TRIAGE_PARAMS, **(params or {}), text_start=1 + NUM_PATCHES)
        self.info = None
        self.reset()

    def reset(self):
        self.prev, self.kf = None, None

    def unnorm_key(self, name):
        key = name if name in self.vla.norm_stats else f"{name}_no_noops"
        assert key in self.vla.norm_stats, f"no un-normalization key for {name}"
        return key

    def __call__(self, image, task_label, unnorm_key, crop=True):
        image = Image.fromarray(image).convert("RGB")
        if crop:
            image = center_crop(image)
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        st = dict(self.params, prev=self.prev, kf=self.kf)
        self.vla.language_model.config.triage = st
        action, _ = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False, return_dict_in_generate=True,
                                            output_attentions=False, past_key_values=DynamicCache())
        self.vla.language_model.config.triage = None
        self.prev, self.info = st["out"], st["info"]
        new, dec = st["kf_new"], st["kf_dec"]
        if new is not None and dec:
            fc1 = torch.stack(dec).mean(dim=0)[1:1 + new["emb"].shape[0]]
            if "kept" in new:
                hv = torch.nn.functional.normalize(new["h2"], dim=1)
                drop = torch.ones(fc1.numel(), dtype=torch.bool, device=fc1.device)
                drop[new["kept"]] = False
                if bool(drop.any()):
                    nb = torch.topk(hv[drop] @ hv[new["kept"]].t(), min(4, new["kept"].numel()), dim=1).indices
                    fc1 = fc1.clone()
                    fc1[drop] = fc1[new["kept"]][nb].mean(dim=1)
            self.kf = {"emb": new["emb"], "fc": torch.stack([new["fc0"], fc1])}
        return action
