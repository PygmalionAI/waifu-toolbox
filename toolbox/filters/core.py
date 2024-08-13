from abc import ABC, abstractmethod, abstractstaticmethod

class ToolboxFilter(ABC):
    @abstractmethod
    def __init__(self) -> None:
        raise NotImplementedError("ToolboxFilter should not be directly called!")
    
    @abstractstaticmethod
    def filter_fn(self, example, **kwargs):
        raise NotImplementedError("ToolboxFilter's filter_fn method should not be directly called!")
        
    @abstractstaticmethod
    def requires_init(self) -> bool:
        # Can return either True or False.
        raise NotImplementedError("ToolboxFilter's requires_init method should not be directly called!")
    