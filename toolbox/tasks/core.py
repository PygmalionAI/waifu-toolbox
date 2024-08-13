from abc import ABC
from typing import Optional, Union

from datasets import Dataset, DatasetDict

from ..filters import ToolboxFilter

class ToolboxTask(ABC):
    def __init__(
        self,
        filter_fn: function,
        custom_prompts=Optional[list[str]] = None,
        **kwargs
    ) -> None:
        self.filter_fn = filter_fn
        self.custom_prompts = custom_prompts

    def process(self, dataset: Union[Dataset, DatasetDict]) -> Dataset:
        raise NotImplementedError("ToolboxTask's process function cannot be directly called!")