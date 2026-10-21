"""Causal right-wrist visual history, sharing augmentation with the current frame."""
from giga_train import TRANSFORMS
from .wa_transforms_lerobot_pretrain import WALeRobotTransformsPretrain


@TRANSFORMS.register
class WALeRobotShortMemoryTransforms(WALeRobotTransformsPretrain):
    def __init__(self, history_frames=8, **kwargs):
        super().__init__(**kwargs)
        self.history_frames = int(history_frames)
        if self.history_frames != 8 or self.num_views != 1:
            raise ValueError("This memory configuration requires eight history frames and one view")

    def __call__(self, data_dict):
        out = super().__call__(data_dict)
        images = out["images"]
        if images.shape[0] != self.history_frames + 5:
            raise ValueError(f"Expected eight past and five current/future frames, got {images.shape}")
        # Only past/current frames enter the clean memory encoder. Dataset sampling
        # clamps negative offsets at the start of the SAME episode.
        out["memory_images"] = images[:self.history_frames + 1].clone()
        out["images"] = images[self.history_frames:]
        out["ref_images"] = out["images"].clone()
        out["ref_images"][1:] = 0
        _, masks = self.mask_generator.get_mask(5)
        out["ref_masks"] = masks[None, :, None, None]
        return out
