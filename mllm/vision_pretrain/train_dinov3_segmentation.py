"""DINOv3 defaults for the shared road-lane segmentation trainer."""

from .train_dinov2_segmentation import main


if __name__ == "__main__":
    main(default_vision_model_type="dinov3")
