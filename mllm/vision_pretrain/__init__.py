"""Vision-backbone pretraining utilities for BEV road understanding."""

from .dinov2_segmentation import Dinov2RoadSegmentationModel
from .dinov3_segmentation import Dinov3RoadSegmentationModel

__all__ = ["Dinov2RoadSegmentationModel", "Dinov3RoadSegmentationModel"]
