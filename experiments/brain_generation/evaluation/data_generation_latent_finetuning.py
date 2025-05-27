import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import gc
from tqdm import tqdm
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline
import time
import csv
from pathlib import Path

class GuidedSampler:
    def __init__(self, model_id="benetraco/latent_finetuning", resolution=32, num_inference_steps=999,
                 device="cuda", seed=17844):
        self.device = device
        self.seed = seed
        self.resolution = resolution
        self.generator = torch.manual_seed(seed)
        self.num_inference_steps = num_inference_steps

        self.pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        self.pipe.to(device)

        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.scheduler = self.pipe.scheduler
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder

        self.scheduler.set_timesteps(self.num_inference_steps)

    def _get_embeddings(self, prompt):
        tokens = self.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            return self.text_encoder(**tokens).last_hidden_state

    def sample(self, prompts, guidance_values, images_per_combination=50, batch_size=8,
               base_output_dir="generated_images/latent_finetuning"):
        uncond_emb = self._get_embeddings("")

        all_times = []

        for prompt in prompts:
            text_emb = self._get_embeddings(prompt)
            tag = prompt.split()[0]

            for g in guidance_values:
                outdir = os.path.join(base_output_dir, tag, f"g{g}")
                os.makedirs(outdir, exist_ok=True)

                generation_times = []

                for img_idx in tqdm(range(images_per_combination),
                                    desc=f"{tag} | guidance {g}", leave=False):
                    start_time = time.time()

                    latent = torch.randn(1, 4, self.resolution, self.resolution,
                                         generator=self.generator).to(self.device)
                    latent *= self.scheduler.init_noise_sigma
                    lat = latent.clone()

                    for t in self.scheduler.timesteps:
                        lat = self.scheduler.scale_model_input(lat, t)

                        with torch.no_grad():
                            n_uncond = self.unet(lat, t, encoder_hidden_states=uncond_emb).sample
                            n_text = self.unet(lat, t, encoder_hidden_states=text_emb).sample
                            n = n_uncond + g * (n_text - n_uncond)

                        lat = self.scheduler.step(n, t, lat).prev_sample

                    lat /= self.vae.config.scaling_factor
                    with torch.no_grad():
                        decoded = self.vae.decode(lat).sample.cpu()

                    decoded = (decoded + 1.0) / 2.0
                    decoded = decoded.clamp(0, 1)

                    save_path = os.path.join(outdir, f"image_{img_idx}.png")
                    save_image(decoded[0], save_path)

                    end_time = time.time()
                    generation_times.append(end_time - start_time)

                    del lat, decoded, n, n_uncond, n_text
                    gc.collect()
                    torch.cuda.empty_cache()

                # Save average time to a shared list
                avg_time = sum(generation_times) / len(generation_times)
                all_times.append({
                    "model_id": self.pipe.config._name_or_path,
                    "prompt": prompt,
                    "guidance": g,
                    "images": images_per_combination,
                    "avg_time_sec": round(avg_time, 4)
                })

        # Save to CSV
        Path("generation_times").mkdir(exist_ok=True)
        csv_path = Path("generation_times") / f"{self.pipe.config._name_or_path.replace('/', '_')}_timing.csv"
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model_id", "prompt", "guidance", "images", "avg_time_sec"])
            writer.writeheader()
            writer.writerows(all_times)

        print(f"\nSaved generation times to: {csv_path}")


# Example usage:
if __name__ == "__main__":
    sampler = GuidedSampler(
        # model_id="benetraco/latent_finetuning_encoder",
        # model_id="benetraco/latent_finetuning_scanners",
        model_id="benetraco/latent_finetuning_scanners_healthy",
        resolution=32,
        num_inference_steps=999
    )

    sampler.sample(
        # prompts=["SHIFTS FLAIR MRI", "VH FLAIR MRI", "WMH2017 FLAIR MRI"],
        prompts=["Philips FLAIR MRI", "Siemens FLAIR MRI", "GE FLAIR MRI",],
        # guidance_values=[1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 0.0],
        guidance_values=[1.0, 2.0, 3.0, 5.0, 0.0],
        # guidance_values=[10.0],
        images_per_combination=25,
        # batch_size=4,
        base_output_dir="generated_images/latent_finetuning_scanners_healthy"
    )
