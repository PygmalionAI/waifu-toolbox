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
from ..datasets import RpGuildDataset
from ..utils import (
    clean_message,
    thread_unsalvagable,
    PromptManager
)

LOG = logging.getLogger(__name__)

class RpGuildRoleplayTask(BaseTask):
    '''
    Task to do a roleplay.
    '''
    def __init__(
        self, 
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        remove_ooc: bool = True,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        self.remove_ooc = remove_ooc
        if custom_prompts is None:
            kwargs = {"custom_prompts": SYSTEM_PROMPTS} if custom_prompts is None \
            else {"custom_prompts": custom_prompts}
        self.sys_prompts = PromptManager(**kwargs)
        # PromptManagers for the different writing styles.
        self.style_prompts = {style: PromptManager(custom_prompts=STYLE_PROMPT_MAPPING[style]) for style in WRITING_STYLE_TAGS}
        # PromptManagers for the different genres.
        self.genre_prompts = PromptManager(custom_prompts=GENRE_PROMPTS)
        # PromptManagers for the different time periods.
        self.time_prompts = PromptManager(custom_prompts=TIME_PROMPTS)
        # PromptManagers for the NSFW tags.
        self.nsfw_prompts = PromptManager(custom_prompts=NSFW_PROMPTS)

    def _reset_buffer(self) -> None:
        '''Resets the buffer.'''
        self.previous_author = None
        self.full_post = None
        self.current_kind = TurnKind.USER

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task RpGuildRoleplayTask.")
        for thread in RpGuildDataset():
            # Set up a buffer for keeping track of authors.
            # If two posts are made by the author in a row, chain it. This way
            # we avoid repeated human/model turns, which can confuse the model
            # big time.
            self._reset_buffer()

            # Get rid of any unsalvageable threads.
            if thread.thread_name in BROKEN_THREADS:
                LOG.debug(f"Skipping thread {thread.thread_name} due to being broken.")
                continue

            # Get rid of non-IC threads.
            if thread.thread_type != "IC":
                continue

            # Prune threads with less than 2 messages.
            if len(thread.messages) < 2:
                LOG.debug(f"Skipping thread {thread.thread_name} due to being too short.")
                continue

            # If the thread only has one author, no easy way to tell between human
            # and model turns.
            usernames = set([t.author for t in thread.messages])
            if len(usernames) != 2:
                LOG.debug(f"Skipping {thread.thread_name} that doesn't have 2 authors!")
                continue

            # Build up a dictionary of usernames to replace for privacy reasons.
            usernames = set([message.author for message in thread.messages])
            username_substitutions: dict[str, str] = {}
            for idx, name in enumerate(usernames):
                username_substitutions[name] = "{{char_" + str(idx) + "}}"

            # Generate the system prompt.
            sys_prompt = self.sys_prompts.sample_prompt()
            # Now take the first style prompt it sees.
            for tag in thread.tags:
                if tag in WRITING_STYLE_TAGS:
                    style_prompt = self.style_prompts[tag].sample_prompt()
                    # Fifty-fifty chance to put either the system prompt or the style prompt first.
                    if random.random() < 0.5:
                        sys_prompt = f"{sys_prompt} {style_prompt}"
                    else:
                        sys_prompt = f"{style_prompt} {sys_prompt}"
                    break

            # Generate the genre and time period prompts.
            genre_str, time_str = _combine_tags_into_str(thread.tags)
            if genre_str is not None:
                genre_prompt = self.genre_prompts.sample_prompt()
                sys_prompt = f"{sys_prompt} {genre_prompt}{genre_str}"
            if time_str is not None:
                time_prompt = self.time_prompts.sample_prompt()
                sys_prompt = f"{sys_prompt} {time_prompt}{time_str}"
            # NSFW prompt.
            if "18+" in thread.tags:
                nsfw_prompt = self.nsfw_prompts.sample_prompt()
                sys_prompt = f"{sys_prompt} {nsfw_prompt}"

            # We've finally finished generating the system prompt;
            # now add it to the turns.
            turns = [Turn(utterance=sys_prompt, kind=TurnKind.SYSTEM)]

            # Assign usernames to either the user or the model.
            # Since we've checked beforehand that there's only two authors,
            # we can just assign the first one to the user and the second one
            # to the model.
            user_author = self.previous_author = thread.messages[0].author
            model_author = (usernames - {user_author}).pop()
            roles = {
                user_author: TurnKind.USER,
                model_author: TurnKind.MODEL
            }

            for message in thread.messages:
                # Check the author, if it's *not* the same as the previous
                # author then yield the full turn. This should be done *first*.
                if message.author != self.previous_author and \
                self.full_post is not None:
                    turns.append(
                        Turn(
                            utterance=self.full_post,
                            kind=roles[self.previous_author],
                            # TODO(TG): Assign a proper name.
                            name="TODO"
                        )
                    )
                    self.previous_author = message.author
                    self.full_post = ""

                # Now that we got past the first check, empty the string.
                if self.full_post is None:
                    self.full_post = ""

                # Process the message.
                cleaned_message = clean_message(
                    message.message, 
                    username_substitutions,
                    self.remove_ooc
                )

                self.full_post = (self.full_post + f"\n{cleaned_message}").strip()

            # Yield the final turn.
            final_kind = roles[thread.messages[-1].author]
            turns.append(
                Turn(
                    utterance=self.full_post,
                    kind=final_kind,
                    name="TODO"
                )
            )

            # Sometimes we just have a situation where the HTML cleaning
            # results in a faulty message.
            # If this is the case for every message, just ditch the thread.
            if thread_unsalvagable(turns[1:]):
                LOG.info(f"Skipping {thread.thread_name} due to being deemed 'unsalvagable'!")
                continue

            # Update the system prompt by filling in the template strings.
            turns = self.fill_response_template_strs(turns, generation=self.full_post)

            episode = Episode(
                turns=turns,
                identifier=f"rpguild-{thread.thread_name}"
            )
            if self.should_keep(episode):
                # Passed through filters!
                yield episode

def _combine_tags_into_str(tags: list) -> tuple[str, str]:
    '''Combines tags into a string.'''
    def construct_conjunction(tags: list) -> str:
        '''
        Converts lists of tags into a natural sounding sentence. Works like this:
        Given no tags, return `None`
        Given a list `[x]`, simply return `x`
        Given a list `[x, y]`, return "x and y"
        Given a list `[x, y, z]`, convert it to a string `"x, y, and z"
        '''
        # TODO(TG): Again, I have a feeling there's a better way to do this.
        if len(tags) == 0:
            return
        elif len(tags) == 1:
            return tags[0]
        elif len(tags) == 2:
            return f"{tags[0]} and {tags[1]}"
        elif len(tags) < 2:
            return f"{', '.join(tags[:-1])} and {tags[-1]}"
        
    genre_tags = []
    time_tags = []

    for tag in tags:
        if tag in GENRE_TAGS:
            desc = _GENRE_TO_DESC_MAPPING[tag]
            genre_tags += [desc]
        elif tag in TIME_PERIOD_TAGS:
            desc = _TIME_TO_DESC_MAPPING[tag]
            time_tags += [desc]

    return construct_conjunction(genre_tags), construct_conjunction(time_tags)

# Tags.
WRITING_STYLE_TAGS = ["Free", "Casual", "Advanced"]
GENRE_TAGS = ["Horror", "Sci-Fi", "School", "Tabletop", "Nation", "Arena", "Military", "Fantasy", "Romance", "Slice of Life", "Anime/Manga", "Fandom", "Steampunk", "Superhero"]
TIME_PERIOD_TAGS = ["Western", "Ancient", "Apocalyptic", "Post-Apocalyptic", "Historical", "Medieval", "Modern", "Future"]

SYSTEM_PROMPTS = [
    '''%{Enter|Engage|Enable|Start|Commence} %{fictional RP|roleplay|RP} %{mode|as a task|since your goal is to do that}. {{response_length_str}}.''',
    #
    '''%{You are now in|You have entered|You will now start|You will enter|You're in|You've entered|You'll start in} %{fictional RP|roleplay|RP|conversational RP} mode. %{Drive|Move} the %{roleplay|RP|conversation} forward in chunks. {{response_length_str}}.''',
    #
    # Modified SillyTavern prompt
    '''%{Write|Generate|Predict|Create} the %{next|following|subsequent} %{reply|turn} in a %{fictional|interactive} %{roleplay|RP} %{chat|conversation|convo}. {{response_length_str}}.''',
    #
    '''%{SYSTEM|MODE}: %{conversational roleplay|RP|roleplay mode|RP system engaged}
%{NOTE|ADVISORY|KEEP IN MIND}: {{response_length_str}}''',
    #
    '''%{I am|I'm} %{in|currently in|engaging in|beginning|part of} a %{roleplay|RP|fictional roleplay-like conversation} with %{someone else|other people|a user|another person}.''',
    #
    '''{{response_length_str}}.''',
    #
    '''%{OBJECTIVE|TASK|MISSION|JOB} - %{Conduct|Generate|Enjoy|Be a participant in} a %{roleplay session|RP|fictional roleplay}
%{REMEMBER|NOTE|KEEP IN MIND} - {{response_length_str}}''',
    # Misspellings intentional
    '''%{do|make|have} %{rp adventures|writing|creative roleplay|roleplay|rp}
%{pls|please} %{rember|remember} %{to|too|2} %{b|be} %{engaging|immersive|epic}''',
    #
    '''%{You are|You're} in an %{uncensored|unfiltered} %{never-ending|never ending|neverending|unending|unceasing} %{roleplay|RP|fictional roleplay}.
%{Follow|Obey|Heed|Stick to} %{these|the} %{guidelines|rules|guidelines during your generations|rules set for the RP}:
{{response_length_str}}.'''
]

# Writing style prompts
FREE_PROMPTS = [
    "%{Write|Compose} in a %{short|brief} and informal %{manner|way|writing style}.",
    "Be %{freehand|laid back|informal|casual|relaxed} in terms of %{writing|composition}; don't put too %{much effort|many words} into it.",
    "%{Treat|Take} this as a %{casual|quick|relaxed} %{RP|roleplay} session.",
]

CASUAL_PROMPTS = [
    "%{Treat|Take} %{this|the} %{roleplay|RP} %{somewhat|moderately} seriously, but not too detailed.",
]

ADVANCED_PROMPTS = [
    "%{Write|Compose|Generate|Create|Do} responses with %{heavy|great} detail and make every reply have a long length.",
    "%{Responses|Replies} should be very %{detailed|complex} and should contain multiple paragraphs.",
    "%{Treat|Take} %{this|the|the following} %{roleplay|RP} very seriously; put a lot of effort into %{replies|responses} and make them very long and intricate."
]

# It's incomplete because the script will finish the rest depending on the time period.
TIME_PROMPTS = [
    " %{The|This} %{roleplay|RP} %{is set|takes place} %{in|during} ",
    " The %{time period|setting} of %{this|the} %{roleplay|setting|RP} is ",
    " %{Time period|Setting|Tags}: "
]

GENRE_PROMPTS = [
    " %{Genre|Theme(s)|Tags}: ",
    " The %{type|genre} of %{this|the} %{roleplay|RP} is ",
    " The %{themes|genres} are "
]

NSFW_PROMPTS = [
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be not safe for work|be NSFW|include adult themes|include erotic themes|include 18+ content}",
]

STYLE_PROMPT_MAPPING = {
    "Free": FREE_PROMPTS,
    "Casual": CASUAL_PROMPTS,
    "Advanced": ADVANCED_PROMPTS
}

# Genre keyword prompts
_GENRE_TO_DESC_MAPPING = {
    "Horror": "horror",
    "Sci-Fi": "sci-fi",
    "School": "school life",
    "Tabletop": "tabletop games",
    "Nation": "nation-states",
    "Arena": "fighting",
    "Military": "war and the military",
    "Fantasy": "fantasy",
    "Romance": "romance",
    "Slice of Life": "slice of life",
    "Anime/Manga": "anime/manga",
    "Fandom": "an existing fandom",
    "Steampunk": "steampunk",
    "Superhero": "superheroes"
}

_TIME_TO_DESC_MAPPING = {
    "Western": "the time period of the Wild West",
    "Ancient": "ancient times",
    "Apocalyptic": "the apocalypse",
    "Post-Apocalyptic": "after an apocalypse",
    "Historical": "the past",
    "Medieval": "medieval times",
    "Modern": "modern times",
    "Future": "the future"
}

# At least one thread I saw has either been edited post-scrape or something,
# because the entries just say "cut" and are as a result garbage training data.
# Have a variable to sift out threads which consist of only this nonsense.
BROKEN_THREADS = [
    "SAO: Aincrad (1x1 between"
]
