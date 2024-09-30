import logging
import random
import re

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import SlimOrcaDataset
from ..utils import PromptManager

# Should handle most instances of "You are a(n)... assistant"
ASSISTANT_PATTERN = re.compile(r"^You are a.*?\.\s*")

SUMMARY_PHRASES = [
    "in conclusion",
    "in summary",
    "to sum up",
]

LOG = logging.getLogger(__name__)

class SlimOrcaInstructionFollowingTask(BaseTask):
    '''SlimOrca instruction following task.'''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        max_examples: int = 20000,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        # If no custom prompts, use the generic "assistant" prompts
        kwargs = {"generic_prompts": "assistant"} if custom_prompts is None \
            else {"custom_prompts": custom_prompts}
        self.prompts = PromptManager(**kwargs)
        self.max_examples = max_examples

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task SlimOrcaInstructionFollowingTask.")
        for idx, example in enumerate(SlimOrcaDataset()):
            if idx >= self.max_examples:
                break

            # If there's obvious GPT-like summary phrases, have a chance of removing them.
            if any(phrase in example.response.lower() for phrase in SUMMARY_PHRASES):
                if random.random() < 0.5:
                    continue

            system_prompt = self.prompts.sample_prompt()
            # Remove the default "you are an AI assistant" instruction which is
            # typically in the first sentence of an OpenOrca system prompt
            additional_instructions = re.sub(ASSISTANT_PATTERN, "", example.system_prompt)
            if additional_instructions != "":
                system_prompt += f"\n{additional_instructions}"
            turns: list[Turn] = [
                Turn(
                    utterance=system_prompt,
                    kind=TurnKind.SYSTEM,
                ),
                Turn(
                    utterance=example.question,
                    kind=TurnKind.USER,
                ),
                Turn(
                    utterance=example.response,
                    kind=TurnKind.MODEL,
                ),
            ]

            # Run through the filters.
            episode = Episode(turns=turns, identifier=f"slimorca-instruct-{idx}")
            if self.should_keep(episode):
                # Passed through filters!
                yield episode
