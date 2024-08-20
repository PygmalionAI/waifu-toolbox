from typing import Type

from ..core import BaseTask
from .aesir_roleplay import AesirRoleplayTask
from .aidungeon_text_adventure import AiDungeonTextAdventureTask
from .airoboros1_guess_instruction import Airoboros1GuessInstructionTask
from .airoboros1_instruction_following import Airoboros1InstructionFollowingTask
from .aitown_roleplay import AiTownRoleplayTask
from .capybara_instruction_following import CapybaraInstructionFollowingTask
from .characterai_roleplay import CharacterAiRoleplayTask
from .claude_multiround_instruct import ClaudeMultiroundInstructTask
from .clubfloyd_text_adventure import ClubFloydTextAdventureTask
from .dolly_guess_instruction import DollyGuessInstructionTask
from .gemstruct_chat import GemstructChatTask
from .limarp_roleplay import LimaRpRoleplayTask
from .magnum_mixed_tasks import MagnumMixedPurposeTask
from .mcstories_writing import McStoriesWritingTask
from .norobots_guess_instruction import NoRobotsGuessInstructionTask
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
        Airoboros1GuessInstructionTask,
        Airoboros1InstructionFollowingTask,
        AiTownRoleplayTask,
        CapybaraInstructionFollowingTask,
        CharacterAiRoleplayTask,
        ClaudeMultiroundInstructTask,
        ClubFloydTextAdventureTask,
        DollyGuessInstructionTask,
        GemstructChatTask,
        LimaRpRoleplayTask,
        MagnumMixedPurposeTask,
        McStoriesWritingTask,
        NoRobotsGuessInstructionTask,
        NoRobotsInstructionFollowingTask,
        OpenCaiRoleplayTask,
        PygClaudeRoleplayTask,
        RpForumsRoleplayTask,
        SlimOrcaInstructionFollowingTask,
        TeatimeRoleplayTask,
    ]
}
