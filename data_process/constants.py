import json
from enum import Enum

class Role(Enum):
    System = "system"
    User = "user"
    Assistant = "assistant"
    SystemPrompt = "You are a road-map reconstruction assistant designed to process BEV (Bird's Eye View) images generated from LiDAR data.\nPredict the complete road map from the current patch in the BEV image.\nReturn only valid JSON in the required schema.\nDo not output markdown fences or extra explanation.\nKeep all coordinates in the patch-local coordinate system."
    UserPrompt1 = "<image>\n Your task is to:\nPlease tell me how many lanes in this image."
    UserPrompt2 = "<image>\n Please extract the centerline coordinates of all lanes in the image."
    
    # SystemPrompt = "You are a road-map reconstruction assistant for satellite-image patches.\nPredict the complete road map in the current patch from the satellite image.\nReturn only valid JSON in the required schema.\nDo not output markdown fences or extra explanation.\nKeep all coordinates in the patch-local coordinate system."
    # UserPrompt = "<image>\nPlease construct the complete road map in the current satellite patch."

class EndpointType(Enum):
    START = "start"
    END = "end"
    CUT = "cut"

