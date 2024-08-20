import logging

from collections import Counter
from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import MagnumDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

class MagnumMixedPurposeTask(BaseTask):
    '''
    Magnum consists of different types of tasks, but share a common structure.
    '''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        replace_claude_prompt: bool = False,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        # If no custom prompts, use the generic "assistant" prompts
        # when a system prompt isn't supplied.
        kwargs = {"generic_prompts": "assistant"} if custom_prompts is None \
            else {"custom_prompts": custom_prompts}
        self.prompts = PromptManager(**kwargs)
        self.file_counter = Counter()
        self.replace_claude_prompt = replace_claude_prompt

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task MagnumMixedPurposeTask.")
        for example in MagnumDataset():
            # Either the system prompt is provided or we generate one.
            conv = example.conversation
            if conv[0].role == "system":
                sys_prompt = conv[0].message
                conv = conv[1:]
            else:
                sys_prompt = self.prompts.sample_prompt()

            # Check if Claude is mentioned in the system prompt - if so, replace it
            # with a generic prompt.
            if self.replace_claude_prompt and "Claude" in sys_prompt:
                sys_prompt = self.prompts.sample_prompt()
            
            turns: list[Turn] = [
                Turn(
                    utterance=sys_prompt,
                    kind=TurnKind.SYSTEM,
                )
            ]
            episode_id = f"magnum-{example.id}" if example.id is not None \
                else f"magnum-{example.filename}-{self.file_counter[example.filename]}"
            self.file_counter[example.filename] += 1

            # Go through the rest of the conversation and append turns.
            for message in conv:
                turns.append(Turn(
                    utterance=message.message.strip(),
                    kind=TurnKind.USER if message.role == "human" else TurnKind.MODEL,
                ))

            # Pass through the filters.
            episode = Episode(turns=turns, identifier=episode_id)
            if self.should_keep(episode):
                yield episode

            