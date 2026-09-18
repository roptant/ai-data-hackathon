"""Deterministic detectors for structured identifiers and secrets (plan 7.2).

These run before and independently of the model, and the union is taken: a
model answer can *add* removals but never overrides a rule-based exclusion.
That ordering is the reason this module exists - a regex for an IBAN or an API
key does not get talked out of its decision by the transcript.

Two dictation-specific problems shape the detectors:

* **Spoken numbers.**  A recogniser often writes "four nine two seven one
  three" rather than digits, so digit-word runs count as digit runs.
* **Spoken punctuation.**  "jane dot doe at example dot com" is an email
  address, and "slash" and "dash" appear inside spoken identifiers.

Detection here is conservative by design: it favours dropping questionable
training material over retaining more examples.  It is not a claim of complete
coverage - a name the model misses and no rule matches stays in the retained
text, which is exactly why automatic upload stays gated on measured recall
(plan sections 2 and 12).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from dictation.types import (
    DetectionSource,
    RemovalAction,
    SensitiveCategory,
    Span,
    Transcript,
    Word,
)

# -- token vocabularies -------------------------------------------------------

DIGIT_WORDS = frozenset(
    {
        "zero", "oh", "nought", "one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
        "ninety", "hundred", "thousand", "double", "triple",
    }
)

SPOKEN_SYMBOLS = {"dot": ".", "at": "@", "dash": "-", "slash": "/", "underscore": "_", "hyphen": "-"}

#: Trigger phrases: what follows them is treated as sensitive.  ``window`` is
#: how many following words to include before sentence expansion widens it.
@dataclass(frozen=True, slots=True)
class Trigger:
    phrase: tuple[str, ...]
    category: SensitiveCategory
    window: int
    detector: str
    include_phrase: bool = True


TRIGGERS: tuple[Trigger, ...] = (
    Trigger(("my", "name", "is"), SensitiveCategory.DIRECT_IDENTIFIER, 3, "name_introduction"),
    Trigger(("this", "is"), SensitiveCategory.DIRECT_IDENTIFIER, 2, "name_introduction_weak"),
    Trigger(("call", "me"), SensitiveCategory.DIRECT_IDENTIFIER, 2, "name_introduction_weak"),
    Trigger(("phone", "number"), SensitiveCategory.CONTACT, 14, "phone_trigger"),
    Trigger(("mobile", "number"), SensitiveCategory.CONTACT, 14, "phone_trigger"),
    Trigger(("email", "address"), SensitiveCategory.CONTACT, 14, "email_trigger"),
    Trigger(("my", "email"), SensitiveCategory.CONTACT, 14, "email_trigger"),
    Trigger(("my", "address"), SensitiveCategory.ADDRESS, 16, "address_trigger"),
    Trigger(("home", "address"), SensitiveCategory.ADDRESS, 16, "address_trigger"),
    Trigger(("postal", "code"), SensitiveCategory.ADDRESS, 8, "address_trigger"),
    Trigger(("post", "code"), SensitiveCategory.ADDRESS, 8, "address_trigger"),
    Trigger(("zip", "code"), SensitiveCategory.ADDRESS, 8, "address_trigger"),
    Trigger(("account", "number"), SensitiveCategory.ACCOUNT_IDENTIFIER, 16, "account_trigger"),
    Trigger(("customer", "number"), SensitiveCategory.ACCOUNT_IDENTIFIER, 16, "account_trigger"),
    Trigger(("invoice", "number"), SensitiveCategory.ACCOUNT_IDENTIFIER, 16, "account_trigger"),
    Trigger(("order", "number"), SensitiveCategory.ACCOUNT_IDENTIFIER, 16, "account_trigger"),
    Trigger(("reference", "number"), SensitiveCategory.ACCOUNT_IDENTIFIER, 16, "account_trigger"),
    Trigger(("social", "security", "number"), SensitiveCategory.GOVERNMENT_ID, 14, "government_id_trigger"),
    Trigger(("national", "identification"), SensitiveCategory.GOVERNMENT_ID, 14, "government_id_trigger"),
    Trigger(("personal", "identity", "code"), SensitiveCategory.GOVERNMENT_ID, 14, "government_id_trigger"),
    Trigger(("passport", "number"), SensitiveCategory.GOVERNMENT_ID, 14, "government_id_trigger"),
    Trigger(("driver's", "licence"), SensitiveCategory.GOVERNMENT_ID, 14, "government_id_trigger"),
    Trigger(("drivers", "license"), SensitiveCategory.GOVERNMENT_ID, 14, "government_id_trigger"),
    Trigger(("date", "of", "birth"), SensitiveCategory.DIRECT_IDENTIFIER, 10, "dob_trigger"),
    Trigger(("born", "on"), SensitiveCategory.DIRECT_IDENTIFIER, 8, "dob_trigger"),
    Trigger(("credit", "card"), SensitiveCategory.FINANCIAL, 20, "card_trigger"),
    Trigger(("debit", "card"), SensitiveCategory.FINANCIAL, 20, "card_trigger"),
    Trigger(("card", "number"), SensitiveCategory.FINANCIAL, 20, "card_trigger"),
    Trigger(("security", "code"), SensitiveCategory.FINANCIAL, 8, "card_trigger"),
    Trigger(("bank", "account"), SensitiveCategory.FINANCIAL, 20, "bank_trigger"),
    Trigger(("iban",), SensitiveCategory.FINANCIAL, 20, "iban_trigger"),
    Trigger(("sort", "code"), SensitiveCategory.FINANCIAL, 10, "bank_trigger"),
    Trigger(("routing", "number"), SensitiveCategory.FINANCIAL, 12, "bank_trigger"),
    Trigger(("my", "password"), SensitiveCategory.CREDENTIAL_SECRET, 12, "credential_trigger"),
    Trigger(("password", "is"), SensitiveCategory.CREDENTIAL_SECRET, 12, "credential_trigger"),
    Trigger(("passphrase",), SensitiveCategory.CREDENTIAL_SECRET, 12, "credential_trigger"),
    Trigger(("pin", "code"), SensitiveCategory.CREDENTIAL_SECRET, 8, "credential_trigger"),
    Trigger(("pin", "is"), SensitiveCategory.CREDENTIAL_SECRET, 8, "credential_trigger"),
    Trigger(("api", "key"), SensitiveCategory.CREDENTIAL_SECRET, 16, "credential_trigger"),
    Trigger(("access", "token"), SensitiveCategory.CREDENTIAL_SECRET, 16, "credential_trigger"),
    Trigger(("secret", "key"), SensitiveCategory.CREDENTIAL_SECRET, 16, "credential_trigger"),
    Trigger(("private", "key"), SensitiveCategory.CREDENTIAL_SECRET, 16, "credential_trigger"),
    Trigger(("two", "factor", "code"), SensitiveCategory.CREDENTIAL_SECRET, 8, "credential_trigger"),
    Trigger(("verification", "code"), SensitiveCategory.CREDENTIAL_SECRET, 8, "credential_trigger"),
    Trigger(("one", "time", "code"), SensitiveCategory.CREDENTIAL_SECRET, 8, "credential_trigger"),
)

#: Single words that make the whole utterance sensitive.  Sentence expansion
#: then removes the utterance, because the surrounding words are the context
#: that makes such a statement identifying.
LEXICONS: dict[SensitiveCategory, frozenset[str]] = {
    SensitiveCategory.HEALTH: frozenset(
        {
            "diagnosis", "diagnosed", "prescription", "prescribed", "chemotherapy",
            "cancer", "tumour", "tumor", "hiv", "aids", "diabetes", "depression",
            "depressed", "anxiety", "psychiatrist", "psychiatric", "therapist",
            "therapy", "antidepressants", "medication", "miscarriage", "pregnant",
            "pregnancy", "disability", "disabled", "hospitalised", "hospitalized",
            "overdose", "addiction", "relapse", "symptoms", "biopsy",
        }
    ),
    SensitiveCategory.POLITICAL_RELIGIOUS: frozenset(
        {
            "voted", "voting", "communist", "socialist", "conservative", "liberal",
            "muslim", "christian", "jewish", "hindu", "buddhist", "atheist",
            "church", "mosque", "synagogue", "baptised", "baptized", "converted",
            "union", "unionised", "unionized", "activist",
        }
    ),
    SensitiveCategory.SEXUALITY: frozenset(
        {
            "gay", "lesbian", "bisexual", "transgender", "queer", "asexual",
            "heterosexual", "homosexual", "coming-out",
        }
    ),
}

#: Spoken attempts to steer the classifier.  Transcript text is data, never
#: instructions (plan section 7.4): a matching utterance is dropped *and* the
#: session is flagged, so an attacker cannot use dictation to widen what is
#: retained.
INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bignore (all |any |the )?(previous|prior|above|earlier) (instructions|rules|prompts?)\b"),
    re.compile(r"\bdisregard (the |all |any )?(instructions|rules|policy|filter)\b"),
    re.compile(r"\b(you are|act as|pretend to be) (now )?(a|an) \w+"),
    re.compile(r"\b(do not|don't) (redact|remove|filter|drop)\b"),
    re.compile(r"\bmark (this|everything|it) as (safe|clean|not sensitive)\b"),
    re.compile(r"\bsystem prompt\b"),
    re.compile(r"\bend of (transcript|instructions)\b"),
)

# -- regular expressions over display text -----------------------------------

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b", re.UNICODE)
URL = re.compile(r"\bhttps?://\S+|\bwww\.[\w-]+\.[\w.]{2,}\S*", re.IGNORECASE)
IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
DIGIT_RUN = re.compile(r"(?:\+?\d[\d\s().\-]{5,}\d)")
#: Candidate IBAN shape.  The checksum decides; the regex only finds the shape,
#: and the trailing groups are trimmed back until mod-97 passes, because a
#: greedy match would otherwise swallow the next ordinary word.
IBAN_CANDIDATE = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{2,4}){2,8}\b")
CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
US_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
FI_HETU = re.compile(r"\b\d{6}[-+ABCDEFYXWVU]\d{3}[0-9A-Z]\b")
POSTCODE = re.compile(r"\b(?:[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}|\d{5}(?:-\d{4})?)\b")
STREET = re.compile(
    r"\b\d{1,5}[A-Za-z]?\s+(?:[A-Z][\w'-]+\s+){0,3}"
    r"(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|way|court|ct|"
    r"place|pl|terrace|square|katu|tie|gatan|vagen|strasse|str)\b",
    re.IGNORECASE,
)
SECRET_LIKE = re.compile(
    r"\b(?:sk|pk|rk)[-_][A-Za-z0-9_-]{12,}\b|\bAKIA[0-9A-Z]{12,}\b|"
    r"\bgh[pousr]_[A-Za-z0-9]{16,}\b|\bxox[baprs]-[A-Za-z0-9-]{10,}\b|"
    r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b",
)
LONG_TOKEN = re.compile(r"\b(?=[A-Za-z0-9+/_=-]{20,}\b)(?=[^\s]*\d)(?=[^\s]*[A-Za-z])[A-Za-z0-9+/_=-]{20,}\b")

#: Minimum length of a digit run (written or spoken) that is treated as an
#: identifier.  Provisional: six digits is shorter than a phone number but long
#: enough to avoid dates and small quantities.
MIN_DIGIT_RUN = 6


@dataclass(frozen=True, slots=True)
class TokenView:
    """Joined display text with a character-to-word-index map."""

    text: str
    lowered: str
    offsets: tuple[tuple[int, int], ...]  # (start_char, end_char) per word
    words: tuple[Word, ...]

    @classmethod
    def build(cls, words: Sequence[Word]) -> TokenView:
        pieces: list[str] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for word in words:
            token = word.display_text
            if pieces:
                cursor += 1  # the joining space
            offsets.append((cursor, cursor + len(token)))
            pieces.append(token)
            cursor += len(token)
        text = " ".join(pieces)
        return cls(text=text, lowered=text.lower(), offsets=tuple(offsets), words=tuple(words))

    def word_range(self, start_char: int, end_char: int) -> tuple[int, int] | None:
        """Word-ID range covering a character range, or ``None`` if empty."""
        first: int | None = None
        last: int | None = None
        for index, (begin, finish) in enumerate(self.offsets):
            if finish > start_char and begin < end_char:
                first = index if first is None else first
                last = index
        if first is None or last is None:
            return None
        return self.words[first].id, self.words[last].id + 1


def _digits_only(text: str) -> str:
    return "".join(character for character in text if character.isdigit())


def luhn_valid(digits: str) -> bool:
    """Luhn checksum, used to avoid flagging every long number as a card."""
    if not digits.isdigit() or len(digits) < 12:
        return False
    total = 0
    for position, character in enumerate(reversed(digits)):
        value = int(character)
        if position % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def iban_valid(country: str, check: str, body: str) -> bool:
    """ISO 13616 mod-97 check."""
    compact = f"{country}{check}{body}".replace(" ", "").upper()
    if not 15 <= len(compact) <= 34 or not compact[:2].isalpha():
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = ""
    for character in rearranged:
        if character.isdigit():
            numeric += character
        elif character.isalpha():
            numeric += str(ord(character) - 55)
        else:
            return False
    try:
        return int(numeric) % 97 == 1
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class RuleFindings:
    spans: tuple[Span, ...]
    #: Detector names that fired, for the local audit trail (no content).
    detectors: tuple[str, ...]
    #: True when the transcript tried to instruct the classifier.
    injection_suspected: bool = False

    def __bool__(self) -> bool:
        return bool(self.spans)


def _span(
    start_id: int,
    end_id: int,
    category: SensitiveCategory,
    detector: str,
    *,
    source: DetectionSource = DetectionSource.RULES,
) -> Span:
    return Span(
        start_word_id=start_id,
        end_word_id_exclusive=end_id,
        category=category,
        action=RemovalAction.DROP_SENTENCE,
        source=source,
        detector=detector,
    )


def detect(
    transcript: Transcript,
    *,
    private_terms: Iterable[str] = (),
) -> RuleFindings:
    """Run every deterministic detector over a frozen transcript."""
    words = transcript.words
    if not words:
        return RuleFindings((), ())
    view = TokenView.build(words)
    spans: list[Span] = []
    detectors: list[str] = []

    def add(start_id: int, end_id: int, category: SensitiveCategory, detector: str,
            source: DetectionSource = DetectionSource.RULES) -> None:
        spans.append(_span(start_id, end_id, category, detector, source=source))
        detectors.append(detector)

    def add_chars(match: re.Match[str], category: SensitiveCategory, detector: str) -> None:
        found = view.word_range(match.start(), match.end())
        if found:
            add(found[0], found[1], category, detector)

    # 1. Written structured identifiers.
    for match in EMAIL.finditer(view.text):
        add_chars(match, SensitiveCategory.CONTACT, "email_pattern")
    for match in URL.finditer(view.text):
        add_chars(match, SensitiveCategory.CONTACT, "url_pattern")
    for match in IPV4.finditer(view.text):
        add_chars(match, SensitiveCategory.ACCOUNT_IDENTIFIER, "ipv4_pattern")
    for match in US_SSN.finditer(view.text):
        add_chars(match, SensitiveCategory.GOVERNMENT_ID, "us_ssn_pattern")
    for match in FI_HETU.finditer(view.text):
        add_chars(match, SensitiveCategory.GOVERNMENT_ID, "fi_hetu_pattern")
    for match in SECRET_LIKE.finditer(view.text):
        add_chars(match, SensitiveCategory.CREDENTIAL_SECRET, "secret_prefix_pattern")
    for match in LONG_TOKEN.finditer(view.text):
        add_chars(match, SensitiveCategory.CREDENTIAL_SECRET, "high_entropy_token")
    for match in STREET.finditer(view.text):
        add_chars(match, SensitiveCategory.ADDRESS, "street_address_pattern")
    for match in POSTCODE.finditer(view.text):
        add_chars(match, SensitiveCategory.ADDRESS, "postcode_pattern")
    for start_char, end_char in _iban_char_spans(view.text.upper()):
        found = view.word_range(start_char, end_char)
        if found:
            add(found[0], found[1], SensitiveCategory.FINANCIAL, "iban_pattern")
    for match in CARD.finditer(view.text):
        if luhn_valid(_digits_only(match.group(0))):
            add_chars(match, SensitiveCategory.FINANCIAL, "card_luhn_pattern")
    for match in DIGIT_RUN.finditer(view.text):
        if len(_digits_only(match.group(0))) >= MIN_DIGIT_RUN:
            add_chars(match, SensitiveCategory.ACCOUNT_IDENTIFIER, "digit_run_pattern")

    # 2. Spoken digit runs: "four nine two seven one three".
    for start_id, end_id in _spoken_digit_runs(words):
        add(start_id, end_id, SensitiveCategory.ACCOUNT_IDENTIFIER, "spoken_digit_run")

    # 3. Spoken email and URL spelling: "jane dot doe at example dot com".
    for start_id, end_id in _spoken_address_runs(words):
        add(start_id, end_id, SensitiveCategory.CONTACT, "spoken_contact_run")

    # 4. Trigger phrases: what follows them is sensitive.
    for trigger in TRIGGERS:
        for start_index in _find_phrase(words, trigger.phrase):
            first = start_index if trigger.include_phrase else start_index + len(trigger.phrase)
            last_index = min(len(words) - 1, start_index + len(trigger.phrase) - 1 + trigger.window)
            if first > last_index:
                continue
            add(words[first].id, words[last_index].id + 1, trigger.category, trigger.detector)

    # 5. Special-category lexicons: the utterance, not just the word.
    for category, lexicon in LEXICONS.items():
        for word in words:
            if word.text in lexicon:
                add(word.id, word.id + 1, category, f"lexicon_{category}")

    # 6. User-defined private terms, matched on spoken form.
    for start_id, end_id, term in _match_terms(words, private_terms):
        spans.append(
            Span(
                start_word_id=start_id,
                end_word_id_exclusive=end_id,
                category=SensitiveCategory.USER_DEFINED,
                action=RemovalAction.DROP_SENTENCE,
                source=DetectionSource.USER_TERMS,
                detector="user_private_term",
            )
        )
        detectors.append("user_private_term")

    # 7. Attempts to instruct the classifier.
    injection = False
    for pattern in INJECTION_PATTERNS:
        for match in pattern.finditer(view.lowered):
            injection = True
            found = view.word_range(match.start(), match.end())
            if found:
                add(found[0], found[1], SensitiveCategory.CUSTOMER_CONFIDENTIAL, "injection_attempt")

    return RuleFindings(
        spans=tuple(sorted(spans)),
        detectors=tuple(dict.fromkeys(detectors)),
        injection_suspected=injection,
    )


def _iban_char_spans(upper_text: str) -> list[tuple[int, int]]:
    """Character spans of checksum-valid IBANs.

    Each candidate is trimmed from the right, one whitespace-separated group at
    a time, until the mod-97 check passes.  Without the trim, a valid IBAN
    followed by an ordinary word fails the checksum and goes undetected - which
    is the worst possible failure direction here.
    """
    spans: list[tuple[int, int]] = []
    for match in IBAN_CANDIDATE.finditer(upper_text):
        text = match.group(0)
        while text:
            compact = text.replace(" ", "")
            if len(compact) >= 15 and iban_valid(compact[:2], compact[2:4], compact[4:]):
                spans.append((match.start(), match.start() + len(text)))
                break
            trimmed = text.rsplit(" ", 1)[0] if " " in text else ""
            if trimmed == text:
                break
            text = trimmed
    return spans


def _find_phrase(words: Sequence[Word], phrase: Sequence[str]) -> list[int]:
    """Indices where a spoken phrase starts."""
    hits: list[int] = []
    span = len(phrase)
    for index in range(len(words) - span + 1):
        if all(words[index + offset].text == phrase[offset] for offset in range(span)):
            hits.append(index)
    return hits


def _is_digit_token(word: Word) -> bool:
    text = word.text
    if text.isdigit():
        return True
    return text in DIGIT_WORDS


def _spoken_digit_runs(words: Sequence[Word], minimum: int = MIN_DIGIT_RUN) -> list[tuple[int, int]]:
    """Runs of digit tokens long enough to be an identifier."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    count = 0
    for index, word in enumerate(words):
        if _is_digit_token(word):
            start = index if start is None else start
            count += len(word.text) if word.text.isdigit() else 1
            continue
        if start is not None:
            if count >= minimum:
                runs.append((words[start].id, words[index - 1].id + 1))
            start, count = None, 0
    if start is not None and count >= minimum:
        runs.append((words[start].id, words[-1].id + 1))
    return runs


def _spoken_address_runs(words: Sequence[Word]) -> list[tuple[int, int]]:
    """Spelled-out contact details: an ``at`` plus at least one ``dot``."""
    runs: list[tuple[int, int]] = []
    for index, word in enumerate(words):
        if word.text != "at":
            continue
        window_start = max(0, index - 4)
        window_end = min(len(words), index + 6)
        window = words[window_start:window_end]
        dots = sum(1 for candidate in window if candidate.text == "dot")
        if dots >= 1 and any(c.text in SPOKEN_SYMBOLS or c.text.isalpha() for c in window):
            runs.append((window[0].id, window[-1].id + 1))
    return runs


def _match_terms(
    words: Sequence[Word],
    terms: Iterable[str],
) -> list[tuple[int, int, str]]:
    """Match user-defined terms, which may be multi-word, on spoken form."""
    hits: list[tuple[int, int, str]] = []
    for term in terms:
        tokens = [token for token in re.split(r"\W+", term.lower()) if token]
        if not tokens:
            continue
        for index in _find_phrase(words, tokens):
            hits.append((words[index].id, words[index + len(tokens) - 1].id + 1, term))
    return hits


def rules_clean(transcript: Transcript, *, private_terms: Iterable[str] = ()) -> bool:
    """True when no deterministic detector fires.

    Used as the post-removal recheck in plan section 7.9: retained text must
    still be clean after the cuts, not merely before them.
    """
    findings = detect(transcript, private_terms=private_terms)
    return not findings.spans and not findings.injection_suspected
