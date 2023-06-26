import torch
import torch.nn as nn

import open_clip
from vector_quantize_pytorch import ResidualVQ


class VisualRVQ(nn.Module):

    def __init__(
            self,
            vision_encoder: str = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            dim: int = 768,
            num_quantizers: int = 64,
            codebook_size: int = 4096,
            use_cosine_sim: bool = True,
            kmeans_init: bool = True,
            kmeans_iters:int = 30,
            threshold_ema_dead_code: int = 2,
            orthogonal_reg_weight: float = 10,
            commitment_weight: float = 1,
            sync_codebook: bool = False
    ):
        super().__init__()

        self.vision_encoder, _, self.preprocess = open_clip.create_model_and_transforms(vision_encoder)
        self.rvq = ResidualVQ(
            dim=dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            use_cosine_sim=use_cosine_sim,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            threshold_ema_dead_code=threshold_ema_dead_code,
            orthogonal_reg_weight=orthogonal_reg_weight,
            commitment_weight=commitment_weight,
            sync_codebook=sync_codebook
        )

    
    def forward(self, images, return_all_codes=False):
        images = self.preprocess(images)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.vision_encoder.encode_image(images)
        
        # quantized, indices, commit_loss, all_codes (optional)
        # (batch, seq, dim), (batch, seq, quantizer), (batch, quantizer)
        return self.vq(image_features, return_all_codes)
