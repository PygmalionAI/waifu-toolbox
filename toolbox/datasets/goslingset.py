import json
import logging
import os

from typing import Generator

from .common import MessageAndRole, MessageAndRoleConversation
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

class GoslingDataset(BaseDataset[MessageAndRole]):
    '''
    A small, tiny hand-compiled dataset.
    Won't make a dent in any training dataset in terms of size,
    but I don't really care. It's a fun dataset.
    '''
    def __iter__(self) -> Generator[MessageAndRoleConversation, None, None]:
        root_path = get_path_for("gosling")
        file_path = os.path.join(root_path, "gosling_eros.jsonl")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                conversation = [
                    MessageAndRole(
                        message=m["value"],
                        role=m["from"]
                    ) for m in row["conversations"]
                ]
                yield MessageAndRoleConversation(conversation=conversation)
