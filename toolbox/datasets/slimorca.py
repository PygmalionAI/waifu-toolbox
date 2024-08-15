import logging
import os

from dataclasses import dataclass
from typing import Generator

import pandas as pd

from ..core import BaseDataset
from ..utils import get_path_for

LOG = logging.getLogger(__name__)

@dataclass
class SlimOrcaEntry:
    system_prompt: str
    question: str
    response: str

class SlimOrcaDataset(BaseDataset[SlimOrcaEntry]):
    '''
    A cleaned, deduplicated version of OpenOrca.
    Can be found at https://huggingface.co/datasets/Open-Orca/slimorca-deduped-cleaned-corrected
    The "v4" branch will actually use datasets and all the benefits for it,
    but it's faster to just download the parquet right now than to rewrite the entire toolbox again.
    '''
    def __iter__(self) -> Generator[SlimOrcaEntry, None, None]:
        root_path = get_path_for("slimorca")
        file_path = os.path.join(root_path, "cgato_SlimOrcaDedupCleaned_train.parquet")
        df = pd.read_parquet(file_path)
        for _, row in df.iterrows():
            # It's in ShareGPT format, but there's always only three turns - system, user, model.
            entry = row["conversations"]
            yield SlimOrcaEntry(
                system_prompt=entry[0]['value'],
                question=entry[1]['value'],
                response=entry[2]['value']
            )
