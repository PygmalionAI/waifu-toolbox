'''
Common dataclasses for use in multiple datasets.
'''

from dataclasses import dataclass

@dataclass
class AlpacaLikeDataInstance:
    instruction: str
    input: str
    output: str

@dataclass
class MessageAndRole:
    message: str
    role: str

@dataclass
class MessageAndRoleConversation:
    conversation: list[MessageAndRole]

@dataclass
class MessageAndRoleConversationWithId:
    conversation: list[MessageAndRole]
    id: str

@dataclass
class MessageWithHumanBool:
    message: str
    is_human: bool

@dataclass
class SimpleReplyDataInstance:
    prompt: str
    generation: str
