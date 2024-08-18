import logging
import random

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import LimaRpDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

class LimaRpRoleplayTask(BaseTask):
    '''
    Roleplay task with the LIMARP dataset.
    '''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        # LIMARP already comes prepackaged with system prompts, so we only need to
        # instantiate the PromptManager if custom prompts are provided.
        self.prompts = None
        if custom_prompts is not None:
            kwargs = {"custom_prompts": custom_prompts}
            self.prompts = PromptManager(**kwargs)

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task LimaRpRoleplayTask.")
        for conversation in LimaRpDataset():
            # Get the system prompt from the first message in the conversation
            # or generate one if provided.
            sys_prompt = conversation.conversation[0].message if self.prompts is None \
                else self.prompts.sample_prompt()
            turns: list[Turn] = [
                Turn(
                    utterance=sys_prompt,
                    kind=TurnKind.SYSTEM,
                )
            ]
            # Go through the rest of the conversation and append turns.
            for message in conversation.conversation[1:]:
                turns.append(Turn(
                    utterance=message.message,
                    kind=TurnKind.USER if message.role == "human" else TurnKind.MODEL,
                ))
            # Pass through the filters.
            episode = Episode(turns=turns, identifier=f"limarp-rp-{conversation.id}")
            if self.should_keep(episode):
                yield episode
            