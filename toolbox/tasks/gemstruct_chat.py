import logging
import re
import string

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import GemstructDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

NONSPACED_ELLIPSES_PATTERN = re.compile(r"(\S)(\.{3,}|…)(\S)")

class GemstructChatTask(BaseTask):
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        use_llama_outputs: bool = False,
        capital_threshold: float = 0.7,
        **kwargs
    ) -> None:
        '''
        Task to chat with personas as part of the Gemstruct dataset.
        Special args:
        - use_llama_outputs: whether to use any conversations that were generated with Llama-3.
        - Capital threshold: the threshold for the proportion of capitalized words in a message;
            the average proportion of capitalized words in a bot message must be below this threshold
        '''
        super().__init__(filters=filters)
        self.custom_prompts = custom_prompts
        if self.custom_prompts is not None:
            self.prompts = PromptManager(**kwargs)
        else:
            self.prompts = None
        self.use_llama_outputs = use_llama_outputs
        self.capital_threshold = capital_threshold

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.debug("Processing data for task GemstructChatTask.")
        for idx, chat in enumerate(GemstructDataset()):
            conversation = chat.conversation
            # NOTE(TG): Technically, any model that is trained on even a little bit of Llama-3 data
            # must follow the Llama-3 license. If we really don't wanna do that,
            # we can skip conversations generated with Llama-3. Just to be safe/sure.
            # It's only about ~15% of the data anyway.
            if not self.use_llama_outputs and "llama" in chat.model_name:
                LOG.debug(f"Skipping conversation gemstruct-{idx} because it was generated with Llama-3!")
                continue

            # Handle possible custom system prompts.
            sys_prompt = chat.conversation[0].message if self.custom_prompts is None \
                else self.prompts.sample_prompt()
            conversation = conversation[1:]

            # If the last turn is from the user, drop it.
            if conversation[-1].role == "user":
                conversation = conversation[:-1]

            # Some entries are borked. If no messages from the bot at all, drop it.
            if not any(message.role == "assistant" for message in conversation):
                LOG.debug(f"Skipping conversation gemstruct-{idx} because there are no bot messages!")
                continue

            # And if there's blank messages, trim the conversation to the last non-blank message.
            blank_msg_idx = next((i for i, message in enumerate(conversation) if message.message.strip() == ""), None)
            if blank_msg_idx is not None:
                conversation = conversation[:blank_msg_idx]

            # If less than three messages in the *total* conversation,
            # including system prompt (which was already trimmed from `conversation`, hence `< 2``),
            # drop it.
            if len(conversation) < 2:
                LOG.debug(f"Skipping conversation gemstruct-{idx} because it has less than three non-blank messages!")
                continue

            # Check if the average proportion of capitalized words in the bot messages is below the threshold.
            if self.capital_threshold < 1:
                all_bot_messages = [message.message for message in conversation if message.role == "assistant"]
                avg_capital_proportion = sum(
                    sum(1 for c in message if c in string.ascii_uppercase) / len(message.split())
                    for message in all_bot_messages 
                ) / len(all_bot_messages)
                if avg_capital_proportion >= self.capital_threshold:
                    LOG.debug(f"Skipping conversation gemstruct-{idx} because the average proportion of capitalized words in the bot messages is too high!")
                    continue

            turns = [
                Turn(
                    utterance=sys_prompt,
                    kind=TurnKind.SYSTEM,
                )
            ]

            for message in conversation:
                utterance = message.message.strip()
                # Get rid of non-standard ellipses and ellipses which aren't spaced properly.
                utterance = NONSPACED_ELLIPSES_PATTERN.sub(r"\1... \3", utterance)
                utterance = utterance.replace("…", "...")
                # Non-standard quotation marks and apostrophes too.
                utterance = utterance.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
                turns.append(Turn(
                    utterance=utterance,
                    kind=TurnKind.USER if message.role == "user" else TurnKind.MODEL,
                ))
            
            episode = Episode(turns=turns, identifier=f"gemstruct-{idx}")
            if self.should_keep(episode):
                # Passed through filters!
                yield episode
