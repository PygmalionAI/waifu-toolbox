import json
import logging
import os

from dataclasses import dataclass
from typing import Generator

from .common import MessageAndRole
from ..core import BaseDataset
from ..utils import enumerate_files_for

LOG = logging.getLogger(__name__)

# Sometimes a Magnum entry will have an ID, sometimes it won't.
# If it doesn't, we'll just use None and have the filename plus a number as the ID.
@dataclass
class MagnumDataInstance:
    conversation: list[MessageAndRole]
    filename: str
    id: str | None

def create_magnum_instance(log, filename, id=None):
    return MagnumDataInstance(
        conversation=[
            MessageAndRole(
                message=msg["value"],
                role=msg["from"]
            ) for msg in log["conversations"]
        ],
        filename=filename,
        id=id
    )

class MagnumDataset(BaseDataset[str]):
    '''
    A collection of various synthetically-generated datasets from Anthropic's Claude models.
    '''
    def __iter__(self) -> Generator[str, None, None]:
        for path in enumerate_files_for(dataset_name="magnum", file_extension=[".jsonl", ".json"]):
            filename = os.path.basename(path).split(".")[0]
            with open(path, "r", encoding="utf-8") as f:
                if path.endswith(".json"):
                    logs = json.load(f)
                    for log in logs:
                        yield create_magnum_instance(log, filename, log.get("id", None))
                else:
                    for line in f:
                        log = json.loads(line)
                        yield create_magnum_instance(log, filename)
