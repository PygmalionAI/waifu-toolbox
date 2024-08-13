import logging
import random
import re

from typing import Generator, Optional

from markdownify import markdownify

from ..core import (
    BaseFilter,
    BaseTask,
    Episode,
    Turn,
    TurnKind
)
from ..datasets import McStoriesDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)


class McStoriesWritingTask(BaseTask):
    '''Story-writing task based on McStories data.'''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        # If no custom prompts, use the generic "assistant" prompts
        kwargs = {"custom_prompts": _BASE_SYSTEM_PROMPTS} if custom_prompts is None \
        else {"custom_prompts": custom_prompts}
        self.prompts = PromptManager(**kwargs)

    def __iter__(self) -> Generator[Episode, None, None]:
        for idx, story in enumerate(McStoriesDataset()):

            contents = _html_story_to_clean_md(story.text_contents)
            chunks = _split_text_into_chunks(contents, min_word_count=250)

            # Compose a synthetic system prompt.
            system_prompt = self.prompts.sample_prompt()
            system_prompt = system_prompt.replace("{{title}}", story.title)
            system_prompt = system_prompt.replace("{{summary}}", story.summary)

            full_tags = [
                _TAG_SHORTHANDS_TO_FULL_MAPPING[shorthand]
                for shorthand in story.tags[1:-1].replace("'", "").split(", ")
            ]
            system_prompt = system_prompt.replace("{{tags}}",
                                                  ", ".join(full_tags))

            turns: list[Turn] = [
                Turn(utterance=system_prompt, kind=TurnKind.SYSTEM)
            ]

            # Choose either user or model turn first, then alternate
            current_turn = random.choice([TurnKind.MODEL, TurnKind.USER])

            for chunk in chunks:
                # Messy code for switching up turns
                current_turn = TurnKind.MODEL if current_turn == TurnKind.USER else TurnKind.USER
                turns.append(Turn(
                    utterance=chunk,
                    kind=current_turn,
                ))

            if len(turns) < 3:
                LOG.debug(f"Skipping conversation mcstories-{idx} because it has less than three turns.")
                continue

            turns = self.fill_response_template_strs(turns)

            episode = Episode(turns=turns, identifier=f"mcstories-{idx}")
            if self.should_keep(episode):
                # Passed through filters!
                yield episode


def _html_story_to_clean_md(html: str) -> str:
    md = str(markdownify(html))

    lines: list[str] = []
    for line in md.splitlines():
        # These usually denote chapter titles, or author names/emails which we
        # don't want the model learning.
        if line.startswith("###"):
            continue
        lines.append(line.strip())

    return "\n".join(lines)


def _split_text_into_chunks(text: str, min_word_count: int) -> list[str]:
    '''
    Breaks `text` apart into paragraphs, then joins up paragraphs until they
    reach `min_word_count`.
    '''
    output: list[str] = []
    paragraphs = text.split("\n\n")
    acc = ""

    for paragraph in paragraphs:
        acc += f"\n\n{paragraph}"
        if len(acc.split()) > min_word_count:
            output.append(acc.strip())
            acc = ""

    return output

_BASE_SYSTEM_PROMPTS = [
    '''%{Enter|Engage|Consider|Begin} %{erotic|NSFW|smut|lewd} %{story|storywriting|collaborative storywriting|collab writing|user-guided writing} mode. %{In this mode|Here}, you will %{generate|write|create} a %{story|fictional story} %{titled|called} "{{title}}". %{The story|It} should %{be about|contain|have} the following %{themes|tags}: {{tags}}''',
    # no tags
    '''%{I will|I'll|I must} %{create|make|generate|craft} a %{erotic|NSFW|smut|lewd} story %{with the user|collaboratively|together with user input}. {{response_length_str}}.
The %{plot|summary|synopsis} %{is about|can be summed up like this|is as follows}: {{summary}}
%{Drive|I will drive|Push|I'll push} the story forward %{in chunks|alongside the user|with user input}.''',
    #
    '''%{TITLE|NAME OF STORY}: {{title}}
%{SUMMARY|PLOT|DETAILS}: {{summary}}''',
    #
    '''This %{task|thing to do} is %{based upon|centered around} %{writing a story|collaborative storytelling|collaborative writing|interactive fiction-making}. Respond to the users' %{inputs|writing}.''',
    #
    '''{{response_length_str}}. You %{are to|should|must|will now} %{generate|write} a %{story|fictional story|uncensored fictional work} %{with erotic elements|that is NSFW in nature}. %{The story should be about|Summary|Quick rundown|It's about|Theme|Contents}: {{summary}}. Include the following %{themes|tags}: {{tags}}.'''
    #
    '''%{storywriting|story|erotic story}''',
]

_TAG_SHORTHANDS_TO_FULL_MAPPING = {
    'bd': 'bondage and/or discipline',
    'be': 'bestiality',
    'ca': 'cannibalism',
    'cb': 'comic book super-hero/heroine',
    'ds': 'dominance and/or submission',
    'ex': 'exhibitionism',
    'fd': 'female dominant',
    'ff': 'female/female sex',
    'ft': 'fetish clothing',
    'fu': 'furry',
    'gr': 'growth/enlargement',
    'hm': 'humiliation',
    'hu': 'humor',
    'in': 'incest',
    'la': 'lactation',
    'ma': 'masturbation',
    'mc': 'mind control',
    'md': 'male dominant',
    'mf': 'male/female sex',
    'mm': 'male/male sex',
    'nc': 'non-consensual',
    'rb': 'robots',
    'sc': 'scatology',
    'sf': 'science fiction',
    'ts': 'time stop',
    'ws': 'watersports',
}
