import logging
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
from ..datasets import RpForumsDataset, RpType
from ..utils import (
    clean_message,
    thread_unsalvagable,
    PromptManager
)

LOG = logging.getLogger(__name__)

class RpForumsRoleplayTask(BaseTask):
    '''
    Task to continue a roleplay 
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
        self.prompts = PromptManager(**kwargs)
    
    def _reset_buffer(self) -> None:
        '''Resets the buffer.'''
        self.previous_author = None
        self.full_post = None
        self.current_kind = TurnKind.USER

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task RpForumsRoleplayTask.")
        for thread in RpForumsDataset():
            # Set up a buffer for keeping track of authors.
            # If two posts are made by the author in a row, chain it. This way
            # we avoid repeated human/model turns, which can confuse the model
            # big time.
            self._reset_buffer()

            # These threads usually don't contain actual roleplaying.
            if any([
                    x in thread.thread_name.lower() for x in [
                        "ooc", "o.o.c", "character sheet", "character profile",
                        "character list", "character roster"
                    ]
            ]):
                LOG.debug(f"Skipping {thread.thread_name} due to likely being OOC-related!")
                continue

            if len(thread.messages) < 2:
                LOG.debug(f'Skipping {thread.thread_name} with only one message!')
                continue

            # If the thread only has one author, no way to tell between human
            # and model turns.
            usernames = set([t.author for t in thread.messages])
            if len(usernames) != 2:
                LOG.debug(f"Skipping {thread.thread_name} that doesn't have 2 authors!")
                continue

            # Build up a dictionary of usernames to replace for privacy reasons.
            username_substitutions: dict[str, str] = {}
            for idx, name in enumerate(usernames):
                username_substitutions[name] = "{{char_" + str(idx) + "}}"

            # System prompt
            system_prompt = self.prompts.sample_prompt()
            content_type_prompt = PromptManager(
                CONTENT_TYPE_TO_PROMPTS[thread.content_type]
            ).sample_prompt()
            system_prompt = system_prompt.replace("{{content_type_str}}", \
                content_type_prompt)
            
            # Add system prompt to turn
            turns: list[Turn] = [
                Turn(
                    utterance=system_prompt, kind=TurnKind.SYSTEM
                )
            ]

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
                identifier=f"rpforums-{thread.source_file}-{thread.thread_name}"
            )
            if self.should_keep(episode):
                # Passed through filters!
                yield episode

# Constants

SYSTEM_PROMPTS = [
    '''%{Enter|Engage|Enable|Start|Commence} %{fiction writing|fantasy writing|fantasy roleplay|fictional RP|roleplay|RP} %{mode|as a task|since your goal is to do that}. {{content_type_str}}. {{response_length_str}}.''',
    #
    '''You %{are now in|have entered|will now start|will enter} %{fiction writing|fantasy writing|fantasy roleplay|fictional RP|roleplay|RP|conversational RP} mode. Drive the story forward in chunks. {{response_length_str}}.''',
    #
    '''You %{are trained to|will|must|have to|have a job to|are ordered to} %{perform|generate|make} %{storywriting|fiction writing|fantasy writing|fantasy roleplay|fictional roleplay|RP}. Generate %{continuations|a response|an appropriate response} for whatever the user %{gives|types|inputs}. {{response_length_str}}. {{content_type_str}}.''',
    # Modified SillyTavern prompt
    '''%{Write|Generate|Create} the next %{reply|turn} %{in|for} a %{fictional|interactive} %{roleplay|RP} %{chat|conversation}. {{content_type_str}}. {{response_length_str}}.''',
    #
    '''%{SYSTEM|MODE}: %{conversational roleplay|RP|roleplay mode|RP system engaged}
%{NOTE|ADVISORY|KEEP IN MIND}: {{response_length_str}}''',
    #
    '''%{I am|I'm} %{in|currently in|engaging in|beginning} a %{roleplay|RP|fictional roleplay-like conversation} with %{someone else|other people|a user}.''',
    #
    '''{{content_type_str}}. {{response_length_str}}.''',
    #
    '''%{OBJECTIVE|TASK|MISSION|JOB} - %{Conduct|Generate|Enjoy} a %{roleplay session|RP|fictional roleplay}
%{DISCRETION RATING|SAFE FOR WORK?|CONTENT RATING} - {{content_type_str}}
%{REMEMBER|NOTE} - {{response_length_str}}''',
    # Misspellings intentional
    '''%{do|make|have} %{rp adventures|writing|creative roleplay|roleplay|rp}
%{pls|please} %{rember|remember} to %{b|be} %{engaging|immersive|epic}''',
    #
    '''%{You are|You're} in an %{uncensored|unfiltered} %{never-ending|never ending|neverending} %{roleplay|RP|fictional roleplay}.
%{Follow|Obey|Heed|Stick to|Abide by} %{these|the} %{guidelines|rules|guidelines during your generations}:
{{content_type_str}}. {{response_length_str}}.'''
]

SFW_PROMPTS = [
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} must %{be safe for work|be SFW|not include any adult themes|be safe for minors|not include 18+ content|not be 18+|not be NSFW}",
]

MIXED_SFW_NSFW_PROMPTS = [
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} %{may or may not include adult themes|may or may not be NSFW|can include adult themes|could potentially be 18+ content|may contain adult content|may contain erotic themes|may contain 18+ content|may contain NSFW content}",
]

NSFW_PROMPTS = [
    "%{Generations|Your writing|The generated response|Your reply|Generated replies} %{must|should, in nature} %{be not safe for work|be NSFW|include adult themes|include erotic themes|include 18+ content|contain adult content}",
]

WHITESPACE = ["\n", " ", "\t"]

CONTENT_TYPE_TO_PROMPTS: dict[RpType, list[str]] = {
    RpType.RP: SFW_PROMPTS,
    RpType.ERP: NSFW_PROMPTS,
    RpType.MIXED: MIXED_SFW_NSFW_PROMPTS,
}