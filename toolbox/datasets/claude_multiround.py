import json
import logging
import os

from dataclasses import dataclass
from typing import Generator

from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

@dataclass
class ClaudeMultiroundChat:
    conversation: list[dict[str, str]]
    id: str

class ClaudeMultiroundDataset(BaseDataset[ClaudeMultiroundChat]):
    '''
    Logs taken from synthetically-generated instruction chats with Claude.
    https://huggingface.co/datasets/Norquinal/claude_multiround_chat_30k
    '''
    def __iter__(self) -> Generator[ClaudeMultiroundChat, None, None]:
        root_path = get_path_for("claude-multiround")
        file_path = os.path.join(root_path, "claude_multiround_chat_30k.json")

        with open(file_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
            # Go through the logs and simply fetch them
            for round in logs:
                yield ClaudeMultiroundChat(
                    conversation=round["conversations"],
                    id=round["id"],
                )
