#imports
import torch
from diffusers import StableDiffusionInpaintPipeline

class HistogramInpaintPipeline(StableDiffusionInpaintPipeline):
    def __init__(self, *args, hist_conditioning=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hist_conditioning = hist_conditioning  # MLP module

    def _encode_prompt_with_hist(self, prompt, device, num_images_per_prompt, do_classifier_free_guidance, hist=None):
        prompt_embeds = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance
        )

        if hist is not None and self.hist_conditioning is not None:
            hist = hist.to(device)
            hist_embed = self.hist_conditioning(hist).unsqueeze(1)  # (B, 1, 768)
            prompt_embeds = prompt_embeds + hist_embed

        return prompt_embeds

    def __call__(self, prompt, image, mask_image, hist=None, **kwargs):
        # Override to inject histogram conditioning
        self.hist = hist
        return super().__call__(prompt, image=image, mask_image=mask_image, **kwargs)

    def _encode_prompt(self, prompt, *args, **kwargs):
        # Intercept internal call to use hist
        return self._encode_prompt_with_hist(prompt, *args, hist=self.hist, **kwargs)



class HistogramConditioning(torch.nn.Module):
    def __init__(self, hist_dim=32, hidden_dim=128, embed_dim=768):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hist_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, hist):
        return self.mlp(hist)  # shape: (B, embed_dim)
  