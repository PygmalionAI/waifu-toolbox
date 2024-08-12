from typing import Type

from ..core import BaseTask
from .aesir_roleplay import AesirRoleplayTask
from .aidungeon_text_adventure import AiDungeonTextAdventureTask
from .airoboros_instruction_following import AiroborosInstructionFollowingTask
from .aitown_roleplay import AiTownRoleplayTask
from .characterai_roleplay import CharacterAiRoleplayTask
from .norobots_instruction_following import NoRobotsInstructionFollowingTask
from .opencai_roleplay import OpenCaiRoleplayTask
from .pygclaude_roleplay import PygClaudeRoleplayTask
from .rp_forums_roleplay import RpForumsRoleplayTask
from .teatime_roleplay import TeatimeRoleplayTask

# Make this more dynamic later.
NAME_TO_TASK_MAPPING: dict[str, Type[BaseTask]] = {
    cls.__name__: cls for cls in [
        AesirRoleplayTask,
        AiDungeonTextAdventureTask,
        AiroborosInstructionFollowingTask,
        AiTownRoleplayTask,
        CharacterAiRoleplayTask,
        NoRobotsInstructionFollowingTask,
        OpenCaiRoleplayTask,
        PygClaudeRoleplayTask,
        RpForumsRoleplayTask,
        TeatimeRoleplayTask
    ]
}
