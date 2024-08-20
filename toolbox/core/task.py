import logging
from typing import Optional

from .filters import BaseFilter
from .turns import Episode, Turn
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

class BaseTask:
    '''Base task class. Relies on config fed into this task.'''
    def __init__(
        self,
        filters: list[BaseFilter] = [],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        # We don't call BaseTask directly, but put __init__ here to account for
        # config parameters and task-specific filters.
        self.filters = filters
        # "Custom prompts" in this case refers to the ability for users to
        # specify their own prompts in the prompt config.
        self.custom_prompts = None

    def should_keep(self, example: Episode) -> bool:
        '''
        Filtering on a task-specific level as well as a general check to see if
        the Episode has less than 3 turns.
        '''
        # Cutoff for episodes with less than 3 turns.
        if len(example.turns) < 3:
            LOG.debug(f"Episode {example.identifier} skipped because it has less than 3 turns!")
            return False
        
        for filter in self.filters:
            if not filter.should_keep(example):
                return False
        return True
    
    def fill_response_template_strs(
        self,
        turns: list[Turn],
        generation: Optional[str] = None
    ) -> list[Turn]:
        # Update the system prompt by filling in the template strings.
        generation = turns[-1].utterance if generation is None else generation
        turns[0].utterance = PromptManager.fill_response_style_length(
            turns[0].utterance, generation)
        return turns
