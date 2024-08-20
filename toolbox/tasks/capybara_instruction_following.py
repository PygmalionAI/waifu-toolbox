import logging

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import CapybaraDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

class CapybaraInstructionFollowingTask(BaseTask):
    '''Instruction following task based on the Capybara dataset.'''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        # If no custom prompts, use the generic "assistant" prompts
        kwargs = {"generic_prompts": "assistant"} if custom_prompts is None \
            else {"custom_prompts": custom_prompts}
        self.prompts = PromptManager(**kwargs)

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task CapybaraInstructionFollowingTask.")
        for idx, example in enumerate(CapybaraDataset()):
            sys_prompt = self.prompts.sample_prompt()
            turns: list[Turn] = [
                Turn(
                    utterance=sys_prompt,
                    kind=TurnKind.SYSTEM,
                ),
            ]

            for message in example.conversation:
                turns.extend([
                    Turn(
                        utterance=message.prompt.strip(),
                        kind=TurnKind.USER,
                    ),
                    Turn(
                        utterance=message.generation.strip(),
                        kind=TurnKind.MODEL,
                    ),
                ])

            # Run through the filters.
            episode = Episode(turns=turns, identifier=f"capybara-instruct-{idx}")
            if self.should_keep(episode):
                # Passed through filters!
                yield episode
