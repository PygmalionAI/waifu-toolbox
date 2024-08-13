from datasets import Dataset, load_dataset

from core import ToolboxDataset

class Airoboros1Dataset(ToolboxDataset):
    def __init__(self) -> None:
        super().__init__()

    def load(self) -> Dataset:
        dataset = load_dataset("jondurbin/airoboros-gpt4-1.4.1")
        return dataset["train"]