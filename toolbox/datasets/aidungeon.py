import os

from typing import Generator

from ..core import BaseDataset
from ..utils import get_path_for

class AiDungeonDataset(BaseDataset[str]):
    '''
    AI Dungeon's `text_adventures.txt`.
    '''
    def __iter__(self) -> Generator[str, None, None]:
        root_path = get_path_for("ai_dungeon")
        file_path = os.path.join(root_path, "text_adventures.txt")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                yield line
