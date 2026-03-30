from sub_model import *
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            *[TransformerEncoderBlock(emb_size=emb_size, **kwargs) for _ in range(depth)],
            ClassificationHead(emb_size, n_classes)
        )

        
