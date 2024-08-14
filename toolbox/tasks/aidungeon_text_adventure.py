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
from ..datasets import AiDungeonDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)
USER_INPUTS = re.compile(r"^> (.*)$", flags=re.MULTILINE)

class AiDungeonTextAdventureTask(BaseTask):
    '''Text adventure task based on the AI Dungeon dataset.'''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        kwargs = {"custom_prompts": SYSTEM_PROMPTS} if custom_prompts is None \
        else {"custom_prompts": custom_prompts}
        self.prompts = PromptManager(**kwargs)
        self.dm_names = ["Dungeon Master", "Game Master", "Text Adventure Bot", "AI Dungeon"]

    def _convert_story_to_turns(self, story: str) -> list[Turn]:
        turns: list[Turn] = []
        # Model always goes first.
        current_kind = TurnKind.MODEL

        story = story.replace("<|startoftext|>", "")
        story = story.replace("<|endoftext|>", "").strip()
        # Remove excess newlines.
        story = re.sub(r"\n{3,}", "\n\n", story)

        # Split by user inputs.
        splits = USER_INPUTS.split(story)
        for split in splits:
            if split.strip() == "":
                continue
            name = random.choice(["User", "Player", "You"]) if current_kind == TurnKind.USER else random.choice(self.dm_names)
            turns.append(
                Turn(
                    utterance=split.strip(),
                    kind=current_kind,
                    name=name,
                )
            )
            # Switch the kind
            current_kind = TurnKind.USER if current_kind == TurnKind.MODEL else TurnKind.MODEL

        return turns
        
    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task AiDungeonTextAdventureTask.")
        idx = 0
        current_story = ""

        for line in AiDungeonDataset():
            #print(line)
            if line.rstrip().endswith("<|endoftext|>"):
                # Started a new story, handle the previous one.
                turns = self._convert_story_to_turns(current_story)
                sp = PromptManager(SYSTEM_PROMPTS).sample_prompt()
                turns.insert(0, Turn(utterance=sp, kind=TurnKind.SYSTEM))

                #print(f"Number of story turns: {len(turns)}")
                #print(f"Story turns: {turns}")
                episode = Episode(turns=turns, identifier=f"ai-dungeon-{idx}")

                if self.should_keep(episode):
                    yield episode

                current_story = ""
                idx += 1
            else:
                # Continuation.
                current_story += line

SYSTEM_PROMPTS = [
    '''%{This is|You are|Start|Simulate|You are to simulate|Begin} a text %{adventure|adventure game}. %{In this game|In this adventure|Here}, %{the user|I} will issue commands in first person, and you are to %{proceed|continue|continue the game|advance the game|advance the story|continue the adventure} accordingly.'''
    '''The AI is a %{dungeon master|DM}. Its %{goal|purpose} is to play with the user %{a text adventure game|an interactive fiction game}. The AI will %{drive the plot forward|continue the adventure} whenever the user inputs a prompt.''',
    '''%{I'm|I am|i'm|i am} a tool designed to play a text %{adventure|adventure game|story game|RPG}''',
    '''%{Goal|Objective|Task}: %{Simulate|Conduct|Do|Write} %{a text adventure|an adventure|a CYOA game|a text game|adventure roleplaying game} through text}
Notes: Be %{good|creative|authentic}, %{fun|engaging} and %{detailed|immersive}
Length: {{response_length_str}}''',
    '''%% TEXT %{GAME|ADVENTURE} MODE: %{ACTIVATED|ENGAGED} %%''',
    '''pls be like ai dungeon, roleplay with me an adventure game thx''',
    '''%{Enter|Engage|Consider} %{game|adventure game|text adventure} mode. %{Here|In this mode}, you will respond to %{my|the user's} %{commands|prompts} and drive a %{story|plot} %{forward|forwards}. Commands will be given in %{1st person|first person|my point of view}''',
    "game",
    '''IS_GAME_MASTER = True
if IS_GAME_MASTER:
    execute_%{text_adventure|game|interactive_adventure}(creative=True, advance_plot=True)''',
]
