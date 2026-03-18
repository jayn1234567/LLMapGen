from typing import Dict, List


class MapState:
    def __init__(self) -> None:
        self.global_lines: List[Dict] = []

    def serialize(self) -> List[Dict]:
        return self.global_lines

    def update(self, new_segments: List[Dict]) -> None:
        self.global_lines.extend(new_segments)
