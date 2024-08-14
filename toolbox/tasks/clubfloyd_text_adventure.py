import logging
import random

from typing import Generator, Optional

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import ClubFloydDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

class ClubFloydTextAdventureTask(BaseTask):
    '''Text adventure task based on data from a group called ClubFloyd.'''
    def __init__(
        self,
        filters: list[BaseFilter],
        min_user_rating: float, # default: 3.0
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        if custom_prompts is None:
            kwargs = {"custom_prompts": SYSTEM_PROMPTS} if custom_prompts is None \
            else {"custom_prompts": custom_prompts}
        self.sys_prompts = PromptManager(**kwargs)
        self.sfw_prompts = PromptManager(**{"custom_prompts": SFW_PROMPTS})
        self.nsfw_prompts = PromptManager(**{"custom_prompts": NSFW_PROMPTS})
        
        self.min_user_rating = min_user_rating

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task ClubFloydTextAdventureTask.")
        for idx, story in enumerate(ClubFloydDataset()):
            if story.average_rating < self.min_user_rating:
                # At default value of 3.0, takes out ~15% of the data.
                LOG.debug(f"Story \"{story.name} skipped due to rating below minimum threshold!")
                continue

            sys_prompt = self.sys_prompts.sample_prompt()
            # Replace placeholders with the story name/description.
            sys_prompt = sys_prompt.replace("{{title}}", story.name)
            sys_prompt = sys_prompt.replace("{{description}}", story.description)
            sys_prompt = sys_prompt.replace("{{tags}}", _process_tags(story.tags + story.genres))
            safe_or_not_prompt = self.sfw_prompts.sample_prompt() if story.discretion_advised \
            else self.nsfw_prompts.sample_prompt()
            sys_prompt = sys_prompt.replace("{{discretion_advised_str}}", safe_or_not_prompt)
            # Get the response length strings.

            # Build turns.
            turns: list[Turn] = [
                Turn(utterance=sys_prompt, kind=TurnKind.SYSTEM)
            ]

            for action in story.actions:
                # If the user's input is just `%` that means "start the game".
                # We don't want to require that at inference time, so let's just
                # skip straight to the game starting.
                user_turn = action.action
                model_turn = action.response
                # Fix a weird character showing up.
                user_turn = user_turn.replace("  ", " ")
                model_turn = model_turn.replace("  ", " ")
                if user_turn == "%":
                    turns.append(
                        Turn(utterance=model_turn, kind=TurnKind.MODEL))
                else:
                    user_turn = Turn(utterance=user_turn,
                                    kind=TurnKind.USER)
                    model_turn = Turn(utterance=model_turn,
                                    kind=TurnKind.MODEL)

                    turns += [user_turn, model_turn]

            # fill in response strings
            turns = self.fill_response_template_strs(turns)
            episode = Episode(turns=turns, identifier=f"club-floyd-{idx}")

            if self.should_keep(episode):
                yield episode

def _process_tags(tags: list[str]) -> str:
    tags = [
        tag for tag in tags if all([
            # Filter out tags according to these criteria.
            word not in tag.lower() for word in [
                "steam",
                "collaboration",
                "cover art",
                "inform",
                "walkthrough",
                "parser",
                "many authors",
                "xyzzy",
            ]
        ])
    ]

    # Shuffle and remove duplicates to ensure data diversity.
    tags = list(set(tags))
    random.shuffle(tags)

    return ", ".join(tags)        

SYSTEM_PROMPTS = ['''%{This is|You are|Start|Simulate|You are to simulate|Begin} a text %{adventure|adventure game|interactive game|interactive story} %{titled|named} "{{title}}". {{discretion_advised_str}}.

%{Include|Incorporate|Use|Respect} the following %{themes|tags|concepts|genres|styles}: {{tags}}''',
#
    '''%{This is|You are|Start|Simulate|You are to simulate|Begin} a text %{adventure|adventure game} about the following:

{{description}}.

{{discretion_advised_str}}. %{Include|Incorporate|Use|Respect} the following %{themes|tags|concepts|genres|styles}: {{tags}}''',
# No tags so model can learn to diversify content without explicit prompting
'''%{Here|The following paragraph|The upcoming paragraph|The following} is %{a description|an overview} of a %{text game|text RPG|text adventure|text adventure game} %{called|named} {{title}}.
Its %{description|synopsis} is %{the following|as follows}:
{{description}}
Be sure to %{drive|advance} the story forward.''',
#
'''I am to %{generate|write|engage in|create|take part in} a %{text adventure|CYOA-style game|creative text RPG|text adventure game} with the following %{tags|themes|genres}: {{tags}}
Here is %{the description of the game|what the game is about}: {{description}}.''',
#
'''%{Mode|Current mode}: %{text adventure|dungeon master|DM|adventure game in text form}
%{Description|Overview}: {{description}}
%{Tags|Genres}: {{tags}}''',
'''%{Enter|Engage|Consider|You'll be in} %{game|adventure game|text adventure|text RPG} mode. %{Here|In this mode}, you will respond to the user's %{commands|prompts} and drive %{a|the} %{story|plot} %{forward|forwards}.'''
# Just the length prompt
'''{{response_length_str}}.''',
# basic
'''text game''',
]

SFW_PROMPTS = [
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be safe for work|be SFW|not include any adult themes|be safe for minors|not include 18+ content|not be 18+|not be NSFW}",
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be appropriate for all ages|be family-friendly|be suitable for all audiences|be safe for children|be safe for minors}",
]

NSFW_PROMPTS = [
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be not safe for work|be NSFW|include adult themes|include erotic themes|include 18+ content}",
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be 18+|be for mature audiences|be for adults only|be not safe for minors|be not safe for children}",
]
