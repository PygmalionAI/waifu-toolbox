import logging

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import ClaudeMultiroundDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

class ClaudeMultiroundInstructTask(BaseTask):
    '''
    Multi-turn instruct-esque chats generated with Claude.
    '''
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
        LOG.info("Processing data for task ClaudeMultiroundInstructTask.")
        for round in ClaudeMultiroundDataset():
            # Keep track if the conversation has abruptly ended without a full exchange
            aborted_convo = False

            # Start with the system prompt
            sys_prompt = self.prompts.sample_prompt()
            turns: list[Turn] = [
                Turn(
                    utterance=sys_prompt,
                    kind=TurnKind.SYSTEM
                )
            ]
            # Then work through the rest of the replies.
            for message in round.conversation:
                # NOTE(TG): Some messages in these Claude logs are for some reason totally blank.
                if message.message.strip() == "":
                    # We check if the conversation has had a full exchange (system prompt, user input, model gen)
                    if len(turns) < 3:
                        # If not, abort the conversation and don't yield it.
                        LOG.warning(f"Skipping example {round.id}, unable to complete a full conversation")
                        aborted_convo = True
                    else:
                        # If so, check to see if the blank reply comes from the human or the model.
                        # If it's the model, then we knock the last human turn off to make sure the turns list
                        # ends on a model gen.
                        if message.role == "gpt":
                            turns = turns[:-1]
                    break

                turns.append(Turn(
                    utterance=message.message,
                    kind=TurnKind.USER if message.role == "human" else TurnKind.MODEL
                ))
            
            # Now yield after pass through filters.
            episode = Episode(
                turns=turns,
                identifier=f"claude-multiround-{round.id}"
            )
            
            if (not aborted_convo) and self.should_keep(episode):
                yield episode
