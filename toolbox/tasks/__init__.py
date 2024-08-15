from typing import Type

from ..core import BaseTask
from .aesir_roleplay import AesirRoleplayTask
from .aidungeon_text_adventure import AiDungeonTextAdventureTask
from .airoboros1_instruction_following import Airoboros1InstructionFollowingTask
from .aitown_roleplay import AiTownRoleplayTask
from .characterai_roleplay import CharacterAiRoleplayTask
from .clubfloyd_text_adventure import ClubFloydTextAdventureTask
from .mcstories_writing import McStoriesWritingTask
from .norobots_instruction_following import NoRobotsInstructionFollowingTask
from .opencai_roleplay import OpenCaiRoleplayTask
from .pygclaude_roleplay import PygClaudeRoleplayTask
from .rp_forums_roleplay import RpForumsRoleplayTask
from .slimorca_instruction_following import SlimOrcaInstructionFollowingTask
from .teatime_roleplay import TeatimeRoleplayTask

# Make this more dynamic later.
NAME_TO_TASK_MAPPING: dict[str, Type[BaseTask]] = {
    cls.__name__: cls for cls in [
        AesirRoleplayTask,
        AiDungeonTextAdventureTask,
        Airoboros1InstructionFollowingTask,
        AiTownRoleplayTask,
        CharacterAiRoleplayTask,
        ClubFloydTextAdventureTask,
        McStoriesWritingTask,
        NoRobotsInstructionFollowingTask,
        OpenCaiRoleplayTask,
        PygClaudeRoleplayTask,
        RpForumsRoleplayTask,
        SlimOrcaInstructionFollowingTask,
        TeatimeRoleplayTask
    ]
}
