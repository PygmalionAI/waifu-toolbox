import logging

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import GoslingDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

class GoslingSetMixedPurposeTask(BaseTask):
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ):
        super().__init__(filters=filters)
        self.prompts = PromptManager(**dict(custom_prompts=custom_prompts)) \
            if custom_prompts else None
        
    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.debug("Processing data for task GoslingSetMixedPurposeTask.")
        for idx, entry in enumerate(GoslingDataset()):
            # Basic ShareGPT format. All processing has already been done before.
            sys_prompt = self.prompts.sample_prompt() if self.prompts else entry.conversation[0].message
            turns: list[Turn] = [
                Turn(
                    utterance=sys_prompt,
                    kind=TurnKind.SYSTEM,
                )
            ]
            conversation = entry.conversation[1:]
            for turn in conversation:
                turns.append(
                    Turn(
                        utterance=turn.message,
                        kind=TurnKind.USER if turn.role == "human" else TurnKind.MODEL,
                    )
                )

            episode = Episode(turns=turns, identifier=f"gosling-{idx}")
            if self.should_keep(episode):
                yield episode
