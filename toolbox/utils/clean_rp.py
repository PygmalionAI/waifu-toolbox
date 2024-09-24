import re
import logging

from markdownify import markdownify

from .process import (
    fix_style_and_encoding_issues,
    remove_excessive_newlines,
    remove_links,
    remove_mentions,
    remove_ooc,
    remove_trailing_whitespace_and_bad_lines,
)

CLASS_PATTERN = re.compile(r'(?:<a )?\(?class=\\?".*?(?:(>|$|href=\".*?)("|<\/a>)?)')
MARKDOWN_NOSPACE_PATTERN = re.compile(r"([\w\d])(\*{1,2})([\w\d])")
ONLY_OOC_PATTERN = re.compile(r"^\([^)]*\)\.?$")
REL_PATTERN = re.compile(r"(\[)(.*?)(]\(rel=\))")

LOG = logging.getLogger(__name__)

def clean_message(
    message: str,
    username_subs: dict[str, str],
    clean_ooc_from_msg: bool,
) -> str:
    '''
    Cleans a single message. Best to keep this in its own separate
    function for readability and also so that we can keep it isolated
    from the yielding process, due to having to keep buffers in mind.
    '''
    message = fix_style_and_encoding_issues(message)
    message = remove_bad_html_tags(message)
    message = remove_links(message)

    # Convert to markdown.
    message = str(markdownify(message))
    message = remove_trailing_whitespace_and_bad_lines(message)
    message = fix_markdown(message)

    # Excessive newlines
    message = remove_excessive_newlines(message)

    # Username substitutions need to be done _after_ the HTML has
    # been converted into markdown, otherwise we get escape
    # characters messing things up.
    for name, substitution in username_subs.items():
        message = re.sub(rf"\b{re.escape(name)}\b",
                                substitution, message)
        
    # Remove mentions and OOC if user wants OOC to be purged.
    message = remove_mentions(message)
    if clean_ooc_from_msg:
        message = remove_ooc(message)
    # And weird artifacts
    message = space_before_regex(message)
    message = CLASS_PATTERN.sub("", message).strip()
    message = re.sub(r"\.{4,}", "...", message).replace("…", "...")
    message = message.replace('(align="center">', '')
    message = message.replace('<\/i>', '')
        
    return message

def failed_cleaning(message: str) -> bool:
    '''
    Sometimes markdownify, HTML tag removal and additional processing results
    in a message which has nothing left, likely due to faulty formatting.
    This function attempts to detect those situations, or other situations which
    would result in a blank message.
    '''
    if len(message.strip()) <= 1:
        return True
    # OOC only.
    if ONLY_OOC_PATTERN.search(message) is not None:
        return True
    return False

def fix_markdown(original_message: str) -> str:
    '''
    Fixes markdown issues in the given message.
    TODO(TG): This doesn't fix the asterisks very intelligently. Need to fix this.
    '''
    s = original_message

    # Bold/italics sometimes doesn't have spaces around it after converting from
    # HTML to Markdown for some reason.
    is_opening_asterisk = True
    while (match := MARKDOWN_NOSPACE_PATTERN.search(s)) is not None:
        if is_opening_asterisk:
            s = s[:match.start() + 1] + " " + s[match.start() + 1:]
        else:
            s = s[:match.end() - 1] + " " + s[match.end() - 1:]
        is_opening_asterisk = not is_opening_asterisk

    return s

def remove_bad_html_tags(message: str) -> str:
    '''Cleans up HTML tags we don't want from the given message.'''
    cleaned_message = remove_html_tag(message, "blockquote")
    cleaned_message = remove_html_tag(cleaned_message, "script")

    if "bbImageWrapper" in message:
        # Images are a <div> with some JavaScript to lazy-load them, so we do
        # this behind a guard to reduce false positives just in case.
        cleaned_message = remove_html_tag(cleaned_message, "div")

    return cleaned_message

def remove_html_tag(message: str, tag: str) -> str:
    '''Cleans the given HTML tag from the message.'''
    cleaned_message = message
    cleaning_passes = 0

    while f"<{tag}" in cleaned_message:
        assert cleaning_passes < 4, "Too many cleaning passes, giving up to avoid deadlocking"

        start_idx = cleaned_message.find(f"<{tag}")
        end_idx = cleaned_message.find(f"</{tag}>", start_idx)

        if start_idx == -1 or end_idx == -1:
            LOG.warning("Unbalanced tags found, leaving as-is")
            break

        cleaned_message = cleaned_message[:start_idx] + cleaned_message[
            end_idx + len(f"</{tag}>"):]

    return cleaned_message

def space_before_regex(text: str):
    WHITESPACE = ["\n", " ", "\t"]
    new_text = ''
    last_end = 0
    for match in REL_PATTERN.finditer(text):
        # If the previous character is not a whitespace, add a space before the bracket
        if match.start() > 0 and text[match.start() - 1] not in WHITESPACE:
            new_text += text[last_end:match.start()] + ' ' + match.group(2)
        else:
            new_text += text[last_end:match.start()] + match.group(2)
        
        # If the following character is not a whitespace, add a space after the bracket
        if match.end() < len(text) and text[match.end()] not in WHITESPACE:
            new_text += ' '
        
        last_end = match.end()
    new_text += text[last_end:]  # append the rest of the text after the last match
    
    return new_text

def thread_unsalvagable(turns, threshold=0.5):
    '''
    If the thread is messy enough that we can't salvage a threshold of messages,
    then we just ditch the thread. By default, it's 50%.
    '''
    # Fun fact: True == 1 in Python, so we can just sum up the booleans.
    return sum(failed_cleaning(x.utterance) for x in turns) / len(turns) >= threshold
