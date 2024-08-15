import json
import logging
import os

from typing import Generator

from .common import AlpacaLikeDataInstance
from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

class DollyDataset(BaseDataset[AlpacaLikeDataInstance]):
    '''
    The Dolly instruction dataset from Databricks.
    https://huggingface.co/datasets/databricks/databricks-dolly-15k
    '''
    def __iter__(self) -> Generator[AlpacaLikeDataInstance, None, None]:
        root_path = get_path_for("dolly")
        file_path = os.path.join(root_path, "databricks-dolly-15k.jsonl")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                yield AlpacaLikeDataInstance(
                    instruction=entry["instruction"],
                    input=entry["context"],
                    output=entry["response"]
                )
