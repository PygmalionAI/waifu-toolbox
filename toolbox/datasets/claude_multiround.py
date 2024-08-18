import json
import logging
import os

from typing import Generator

from .common import MessageAndRole, MessageAndRoleConversationWithId
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)


class ClaudeMultiroundDataset(BaseDataset[MessageAndRole]):
    '''
    Logs taken from synthetically-generated instruction chats with Claude.
    https://huggingface.co/datasets/Norquinal/claude_multiround_chat_30k
    '''
    def __iter__(self) -> Generator[MessageAndRoleConversationWithId, None, None]:
        root_path = get_path_for("claude-multiround")
        file_path = os.path.join(root_path, "claude_multiround_chat_30k.json")

        with open(file_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
            # Go through the logs and simply fetch them
            for round in logs:
                conversation = [
                    MessageAndRole(
                        message=msg["value"],
                        role=msg["from"]
                    ) for msg in round["conversations"]
                ]
                yield MessageAndRoleConversationWithId(conversation=conversation, id=round["id"])
