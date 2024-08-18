import json
import logging
import os

from typing import Generator

from .common import MessageAndRole, MessageAndRoleConversationWithId
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)


class LimaRpDataset(BaseDataset[MessageAndRole]):
    '''
    Augmented version of the LIMARP dataset.
    https://huggingface.co/datasets/grimulkan/LimaRP-augmented
    '''
    def __iter__(self) -> Generator[MessageAndRoleConversationWithId, None, None]:
        root_path = get_path_for("limarp")
        file_path = os.path.join(root_path, "LimaRP-augmented.json")

        with open(file_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
            # Go through the logs and simply fetch them
            for rp in logs:
                conversation = [
                    MessageAndRole(
                        message=msg["value"],
                        role=msg["from"]
                    ) for msg in rp["conversations"]
                ]
                yield MessageAndRoleConversationWithId(conversation=conversation, id=rp["id"])
