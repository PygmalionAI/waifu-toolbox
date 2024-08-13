from typing import Optional

from datasets import Dataset

from core import ToolboxTask
from ..filters import ToolboxFilter

class Airoboros1InstructTask(ToolboxTask):
    def __init__(
        self,
        filter_fn: function,
        custom_prompts: Optional[list[str]] = None
    ):
        super().__init__()