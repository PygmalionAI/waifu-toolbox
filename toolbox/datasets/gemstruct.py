import json
import logging
import os

from dataclasses import dataclass
from typing import Generator

from .common import MessageAndRole
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

@dataclass
class GemstructDataInstance:
    conversation: list[MessageAndRole]
    model_name: str

class GemstructDataset(BaseDataset[MessageAndRole]):
    '''
    A dataset of conversations with the Gemma and Llama-3 models.
    '''
    def __iter__(self) -> Generator[GemstructDataInstance, None, None]:
        root_path = get_path_for("gemstruct")
        file_path = os.path.join(root_path, "gemstruct.jsonl") # refuses to load the parquet - sorry! v4 will make us work with HF datasets
        print(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                conversation = [
                    MessageAndRole(
                        message=m["content"],
                        role=m["role"]
                    ) for m in row["conversations"]
                ]
                yield GemstructDataInstance(
                    conversation=conversation,
                    model_name=row["model"]
                )
