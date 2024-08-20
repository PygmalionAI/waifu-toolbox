import logging
import os

from dataclasses import dataclass
from typing import Generator

import ujson

from .common import SimpleReplyDataInstance
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

@dataclass
class CapybaraDataInstance:
    conversation: list[SimpleReplyDataInstance]
    source: str # May or may not be used, but here for posterity

class CapybaraDataset(BaseDataset[SimpleReplyDataInstance]):
    '''
    The Capybara dataset, multi-turn high-quality synthetic conversations.
    https://huggingface.co/datasets/LDJnr/Capybara
    '''
    def __iter__(self) -> Generator[CapybaraDataInstance, None, None]:
        root_path = get_path_for("capybara")
        file_path = os.path.join(root_path, "CapybaraPure_Decontaminated.jsonl")

        with open(file_path, "r", encoding="utf-8") as f:
            for convo in f:
                example = ujson.loads(convo)
                conversation = [
                    SimpleReplyDataInstance(
                        prompt=e["input"],
                        generation=e["output"]
                    ) for e in example["conversation"]
                ]
                yield CapybaraDataInstance(conversation=conversation, source=example["source"])
