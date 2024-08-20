import logging
import re

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import NoRobotsDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

CITATIONS_PATTERN = re.compile(r"\S\[[^\n]\]")

class NoRobotsGuessInstructionTask(BaseTask):
    '''Instruction guessing task based on the No-Robots dataset.'''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        # If no custom prompts, use the generic "assistant" prompts
        kwargs = {"generic_prompts": "gti"} if custom_prompts is None \
            else {"custom_prompts": custom_prompts}
        self.prompts = PromptManager(**kwargs)

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task NoRobotsGuessInstructionTask.")
        for example in NoRobotsDataset():
            sys_prompt = self.prompts.sample_prompt()
            turns: list[Turn] = [
                Turn(
                    utterance=sys_prompt,
                    kind=TurnKind.SYSTEM,
                ),
            ]
            # First, ignore the "Chat" category as that's not a good fit for this task.
            # Or anything else we may catch that has more than two messages.
            if example.category == "Chat" or len(example.conversation) > 2:
                continue

            prompt = example.conversation[0].message
            response = example.conversation[1].message
            # A few of the messages contain citations, which we want to remove.
            prompt = CITATIONS_PATTERN.sub("", prompt).strip()
            response = CITATIONS_PATTERN.sub("", response).strip()

            turns.extend([
                Turn(
                    utterance=response,
                    kind=TurnKind.USER,
                ),
                Turn(
                    utterance=prompt,
                    kind=TurnKind.MODEL,
                ),
            ])

            # Run through the filters.
            episode = Episode(turns=turns, identifier=f"norobots-gti-{example.prompt_id}")
            if self.should_keep(episode):
                # Passed through filters!
                yield episode
