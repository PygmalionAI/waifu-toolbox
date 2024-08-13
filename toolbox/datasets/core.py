from abc import ABC, abstractmethod
from typing import Optional

class ToolboxDataset(ABC):
    # ToolboxDataset may need an initialization but is not often necessary.
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def load(self, hf_name: Optional[str]) -> None:
        raise NotImplementedError("This class should not be utilizing the base load function!")
