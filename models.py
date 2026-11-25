"""Generative models and reward evaluators."""
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, LCMScheduler, StableDiffusionPipeline
from transformers import CLIPModel, CLIPProcessor


class SD15Generator:
    """Stable Diffusion v1.5 generator with DPM-Solver or 4-step LCM-LoRA proxy."""

    MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
    LATENT_SHAPE = (4, 64, 64)

    def __init__(self, use_lcm: bool = False, device: str = "cuda"):
        self.device = device
        self.use_lcm = use_lcm
        print(f"[Init] Loading SD1.5 ({self.MODEL_ID}), use_lcm={use_lcm} ...", flush=True)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.float16, safety_checker=None
        ).to(device)

        if use_lcm:
            self.pipe.load_lora_weights(self.LCM_LORA_ID)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.steps, self.cfg = 4, 1.0
        else:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            self.steps, self.cfg = 20, 7.5
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, prompt: str, x0: np.ndarray):
        lat = torch.from_numpy(x0).float()[None].to(self.device, torch.float16)
        g = torch.Generator(self.device).manual_seed(0)
        return self.pipe(
            prompt,
            latents=lat,
            num_inference_steps=self.steps,
            guidance_scale=self.cfg,
            height=512,
            width=512,
            generator=g,
        ).images[0]


class CLIPReward:
    """CLIP cosine similarity reward scorer."""

    CLIP_ID = "openai/clip-vit-large-patch14"

    def __init__(self, device: str = "cuda"):
        self.device = device
        print(f"[Init] Loading CLIP ({self.CLIP_ID}) ...", flush=True)
        self.model = (
            CLIPModel.from_pretrained(self.CLIP_ID, torch_dtype=torch.float16)
            .to(device)
            .eval()
        )
        self.processor = CLIPProcessor.from_pretrained(self.CLIP_ID)

    @torch.no_grad()
    def score(self, prompt: str, image) -> float:
        inp = self.processor(
            text=[prompt], images=image, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        inp["pixel_values"] = inp["pixel_values"].half()
        out = self.model(**inp)
        sim = torch.nn.functional.cosine_similarity(out.image_embeds, out.text_embeds)
        return float(100.0 * sim.item())


class TaskEvaluator:
    """Combines a generator and a reward model into a black-box objective function."""

    def __init__(self, generator, reward_model):
        self.generator = generator
        self.reward_model = reward_model

    @property
    def latent_shape(self):
        return self.generator.LATENT_SHAPE

    def generate(self, prompt: str, x0: np.ndarray):
        return self.generator.generate(prompt, x0)

    def score(self, prompt: str, image) -> float:
        return self.reward_model.score(prompt, image)

    def make_objective(self, prompt: str):
        """Returns f(x0) -> float for black-box optimizers."""
        def objective(x0: np.ndarray) -> float:
            img = self.generator.generate(prompt, x0)
            return self.reward_model.score(prompt, img)
        return objective