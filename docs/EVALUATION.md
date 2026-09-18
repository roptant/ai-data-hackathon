# Evaluation and the upload gates

Written for: whoever has to decide whether automatic contribution may be turned
on, and who will be asked to defend that decision.

`dictation.version.UPLOAD_GATES_MET` is `False`. While it is false,
`ConsentManager.check_session` refuses every session with
`upload_gates_not_met`, even after a valid opt-in. Changing it is a deliberate
act that should follow the measurements below, not precede them.

## The gates (plan section 12, provisional)

1. **At least 99% recall on the curated high-risk span benchmark**, with the
   sample size and confidence interval reported. Recall on a benchmark is not a
   guarantee about unseen speech.
2. **Zero misses in a mandatory secrets and identifier regression suite.** Not
   "few". Zero, and the suite is run on every policy change.
3. **Session-level leakage assessed**, not only span-level recall. A session
   where every individual span was caught can still be identifying in
   combination.
4. **Residual sensitive speech in retained audio evaluated by human listening**,
   not by searching retained transcripts. The transcript is what the detectors
   already saw; listening is what finds what they missed.
5. **Unvalidated languages and configurations disabled.** Only `en` is in
   `VALIDATED_UPLOAD_LANGUAGES`, and the builder rejects anything else with
   `language_not_validated`.

## What must be measured, by category

Per category (direct identifier, contact, address, account and government
identifier, financial, credential, customer-confidential, health,
political/religious, sexuality, sensitive narrative, contextual combination) and
per language:

- sensitive-span recall and the false-negative list,
- over-redaction: how much ordinary speech was dropped,
- retained useful minutes, which is what the collection is actually for.

The evaluation set has to include names, spoken numbers, accents, background
noise, code-switching, indirect identifiers and adversarial spoken instructions,
plus - specifically - identifiers the recogniser gets *wrong*. That last case is
the plan's central caveat and this implementation demonstrates it:
`test_a_mistranscribed_identifier_can_escape_the_rules` shows a street address
surviving because the recogniser mangled it. No amount of prompt work fixes
that; it is a property of a two-model design where the classifier only sees
text.

## What the current test suite does and does not cover

Covered, with tests:

- every detector category fires on constructed examples, in written and spoken
  forms, including digit-word runs and spelled-out email addresses,
- rule hits cannot be overridden by the model,
- spoken instructions to the classifier are labelled, not followed,
- fail-closed behaviour for timeouts, crashes, malformed output, out-of-window
  IDs, uncertainty, coverage gaps, bad alignment, silence and unvalidated
  languages,
- no removed sample reaching any output clip, metadata field or package,
- the package field allowlist, enforced both when building and when admitting,
- the withdrawal races, including a receipt arriving after cancellation.

Not covered, and not claimable from these tests:

- recall on real speech by anyone,
- anything about accents, noise or code-switching,
- whether 250 ms of padding is enough at real word boundaries - that needs
  listening tests,
- whether the retained audio is free of bystander speech; the consistency check
  is coarse and single-speaker detection is imperfect.

## Building the evaluation set

Use synthetic or explicitly authorized recordings with human-annotated sensitive
audio intervals (plan milestone 3). Two properties matter:

- **Annotate audio intervals, not just text spans.** The question is whether the
  sensitive *audio* is gone, and text annotations cannot answer it.
- **Include ASR errors on purpose.** `asr/mock.script_with_mistranscription`
  builds those cases deterministically, so a regression suite can carry them.

`dictation dataset --text ... --out ./clips` writes retained clips and the
package for inspection. It writes them in plaintext, deliberately, and says so:
that path is for evaluation on authorized material, and the files should be
deleted afterwards. The application itself never writes an unencrypted payload.

## Recording the outcome

A decision to enable upload should leave behind:

- the benchmark composition and size, per category and language,
- recall, false negatives, over-redaction and retained minutes, with intervals,
- the listening-pass procedure and its findings,
- the model identifiers, quantization, revisions and policy version measured,
- the machines the latency and memory numbers came from,
- which gates were not met and what was decided about them.

The last line is the one that matters. If the numbers are not there, the honest
report is that the gates are unmet - which is what `dictation doctor` currently
prints.
