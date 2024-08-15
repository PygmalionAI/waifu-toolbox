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
from ..datasets import DollyDataset
from ..utils import PromptManager

LOG = logging.getLogger(__name__)

CONTEXT_PREFIXES = ["Context: ", "You might want to know this: ", "\nHere's some further information:\n", "Here is the context: ", "Further information: " "", "\n"]

# Dolly's dataset quality isn't great, so we need to explicitly say that.
DOLLY_SYSTEM_PROMPTS = [
    "%{You are|You're|You'll be|You are known as} the Instruction-Guesser. Your %{objective|goal|task|job} %{is that|is} when %{you are|you're} given an answer to %{a question|an inquiry}, %{you will|you'll} %{guess|take a shot at guessing|predict|attempt to match} the instruction that is to go with it. Do not reply with anything else but the instruction. Generated text may be of poor quality.",
    # Diversify formatting a bit
    "%{Name|Your name|Identity}: %{Guesser|Instruction Guesser|The Instruct-Predictor}\nObjective: %{Guess|Predict} instructions upon being given statement and possibly context",
    "%{Enter|Engage|Begin} %{instruction guessing|predictor|instruct-predicting} mode. %{In|Because of|While in} this mode, %{you'll have to|you must|you'll need to} %{guess|predict|take a crack at} what instruction matches with the user's answer.",
    "Given pieces of information, your job is to come up with an instruction that fits with the information. Be %{brisk|brief|straight to the point} in your %{replies|answers|generations}.",
    "%{Welcome to|Consider|You are in} '%{guess|predict} the instruction' mode. Given a response and %{possibly context|potentially some context to go with it|maybe even some additional information related to it}, %{you are|you're} %{tasked with|given the task of|to be} generating the instruction/question that could be applicable to be answered by the response.",
    "instruction %{guessing|flipping|foretelling|predicting} (somewhat poor quality outputs, maybe)",
    "assistant",
    "The Guesser's Persona: Instruction %{Guesser|Predictor|Foreseer} which is tasked with %{predicting|foreseeing|guessing} the %{instruction|instruction and/or question} that goes with a given answer.",
]

DOLLY_USER_PROMPTS = [
    """%{Answer:|Here's an answer for you:|I'm gonna give you you this.|Here's an answer.|The following is a solution to a question or statement:} <INFO> <CONTEXT>\nWhat is %{an|the} instruction that goes with that %{piece|bit} of %{info|information|context}?""",
    """%{Guess|Give me} the %{instruction|instruction I used} given this answer: <INFO> <CONTEXT>""",
    """%{Here is|Here's} %{some information|a piece of text} that corresponds to what %{an assistant|an artificial assistant|a useful digital helper} would generate in response to being given an instruction.
\"<INFO>\" <CONTEXT>
What would have been the %{question|instruction|query} for %{this|that}?""",
    """ok %{here|here u go|lol look}: <INFO>
<CONTEXT>
%{come up with|think of|predict} %{the question|the thing i would've asked you} please""",
    """<INFO> <CONTEXT>"""
]

class DollyGuessInstructionTask(BaseTask):
    '''
    Given an answer and possibly context, task the AI to generate a proper instruction or question for it.
    Heavily inspired by "Guess the Instruction! Flipped Learning Makes Language Models Stronger Zero-Shot Learners"
    Paper: https://arxiv.org/abs/2210.02969 | Github: https://github.com/seonghyeonye/Flipped-Learning/tree/master
    '''
    def __init__(
        self,
        filters: list[BaseFilter],
        custom_prompts: Optional[list[str]] = None,
        **kwargs
    ) -> None:
        super().__init__(filters=filters)
        kwargs = {"custom_prompts": DOLLY_SYSTEM_PROMPTS} if custom_prompts is None \
        else {"custom_prompts": custom_prompts}
        self.system_prompts = PromptManager(**kwargs)
        self.user_prompts = PromptManager(**{"custom_prompts": DOLLY_USER_PROMPTS})

    def __iter__(self) -> Generator[Episode, None, None]:
        LOG.info("Processing data for task DollyGuessTheInstructionTask.")
        for i, entry in enumerate(DollyDataset()):
            turns: list[Turn] = [
                Turn(
                    utterance=self.system_prompts.sample_prompt(),
                    kind=TurnKind.SYSTEM,
                )
            ]
            # Construct user prompt.
            user_prompt = self.user_prompts.sample_prompt()
            # Replace <INFO> with the answer.
            user_prompt = user_prompt.replace("<INFO>", entry.output)
            # Replace <CONTEXT> with the context, if applicable.
            if entry.input != "":
                context = random.choice(CONTEXT_PREFIXES) + entry.input
                user_prompt = user_prompt.replace("<CONTEXT>", context.lstrip())
            else:
                user_prompt = user_prompt.replace("<CONTEXT>", "")

            # Fix excessive whitespace in the instruction.
            instruction = re.sub(r' {2,}', ' ', entry.instruction)

            turns.extend([
                Turn(
                    utterance=user_prompt,
                    kind=TurnKind.USER,
                ),
                Turn(
                    utterance=instruction,
                    kind=TurnKind.MODEL,
                )
            ])

            episode = Episode(turns=turns, identifier=f"dolly-guess-instruction-{i}")
            if self.should_keep(episode):
                yield episode
