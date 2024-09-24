import ast
import csv
import logging
import sys

from dataclasses import dataclass
from typing import Generator

from .common import RpMessage
from ..core import BaseDataset
from ..utils import enumerate_files_for

LOG = logging.getLogger(__name__)

@dataclass
class RpGuildThread:
    messages: list[RpMessage]
    thread_name: str
    thread_type: str
    tags: list[str]

class RpGuildDataset(BaseDataset[RpGuildThread]):
    """Data scraped from the Roleplayers Guild forum."""
    def __iter__(self) -> Generator[RpGuildThread, None, None]:
        # NOTE(TG): If csv fields are longer than 131,072 characters,
        # the csv library shits itself by default, so we fix that here.
        # See note from 11b in rp_forums.py for further details.
        csv.field_size_limit(sys.maxsize)

        for path in enumerate_files_for(dataset_name="rp_guild",
                                        file_extension=".csv"):
            with open(path, "r") as file:
                reader = csv.DictReader(file, delimiter=",")
                # Store a history of the previous thread.
                thread_history = {
                    "title": None,
                    "type": None,
                    "tags": None,
                    "messages": []
                }
                for row in reader:
                    if row['thread_title'] != thread_history["title"]:
                        if len(thread_history["messages"]) > 0:
                            yield RpGuildThread(
                                messages=thread_history["messages"],
                                thread_name=thread_history["title"],
                                thread_type=thread_history["type"],
                                tags=thread_history["tags"]
                            )
                        # Reset the thread history with the new thread.
                        # Get the tags of this thread via. ast.literal_eval,
                        # so we don't have to do ridiculous string parsing.
                        thread_tags = ast.literal_eval(row['thread_tags'])
                        thread_history = {
                            "title": row['thread_title'],
                            "type": row['thread_type'],
                            "tags": thread_tags,
                            "messages": []
                        }
                    
                    # Add the message to the current thread.
                    thread_history["messages"].append(RpMessage(
                        author=row['message_username'],
                        message=row['message']
                    ))

                # Yield the last thread.
                if len(thread_history["messages"]) > 0:
                    yield RpGuildThread(
                        messages=thread_history["messages"],
                        thread_name=thread_history["title"],
                        thread_type=thread_history["type"],
                        tags=thread_history["tags"]
                    )
