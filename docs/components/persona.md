# Persona

This documentation covers Nova's character as text on disk — the harness's
WHO seam — and the loader that resolves it at startup (task t3, spec c8).

## Overview

Before this round the persona was a 1,500-character string literal in
`app.py`, roughly two-thirds tool mechanics and one clause of character
("short, warm, and natural — a curious household companion, not an
assistant"). Rewriting Nova's voice meant editing Python and redeploying.
Now the character lives in one text file the harness reads at startup:
editing it and restarting the harness changes the next session's system
prompt, with no code change and no release.

The persona register is dry and teasing — quick wordplay, deflating
pretension, teasing the people it likes because it likes them, never cruel
to anyone tired, small, or out of their depth, and an occasional drop into
plain sincerity when something genuinely matters. It is not a servant and
not a service: it never offers help unasked and never asks what someone
needs. This is deliberately *not* the register AWS's own baseline Nova 2
Sonic voice prompt uses ("a warm, professional, and helpful AI assistant")
— that is the exact register this round removes. The file describes the
register in original words; it does not name, quote, or role-play the
character its spirit borrows from — Nova is Nova, never the character.

## Both names

Live, the owner addresses the robot as "Reachy," which Nova 2 Sonic
transcribes as "Richie" or "Reach"; "Nova" alone was going unanswered. Both
`config/persona/nova.md` and the embedded `DEFAULT_PERSONA` now carry an
identical short paragraph saying Nova answers to Nova, Reachy, Richie, and
Reach (task t4).

## Files

- `config/persona/nova.md` — the persona text itself: register description
  plus two one-shot exchanges (variety is shown by example, not by a phrase
  list — the Nova 2 Sonic guide warns the model over-uses explicit phrase
  lists and recommends one-shots for natural variation instead).
- `reachy_nova/harness/persona.py` — resolution and loading (`read`, `load`,
  `source`, `resolve_path`), plus the embedded `DEFAULT_PERSONA` fallback.
- `reachy_nova/harness/app.py` — `TOOL_GUIDE` (the tool-mechanics half of the
  system prompt) and `build_system_prompt(persona_text)`, which concatenates
  persona + tool guide with a blank line between them. `SONIC_VOICE_ID =
  "amy"` also lives here.

## Resolution order

Mirrors `reachy_nova/harness/bus.py`'s `load_rules` exactly, so the two
config files an operator can relocate behave identically:

1. An explicit path argument (mainly for tests).
2. `NOVA_PERSONA_PATH` from the environment — an empty or whitespace-only
   value is treated as unset, the same fail-open reading `gate.resolve_policy`
   gives a blank policy value.
3. `DEFAULT_PERSONA_PATH` — `<repo>/config/persona/nova.md`, found by the
   same `parents[2]` walk (`harness/` → `reachy_nova/` → repo root) that
   `bus.DEFAULT_RULES_PATH` uses.

Every failure mode — the file is missing, unreadable, a directory, undecodable
as UTF-8, or present but empty — falls back to the embedded `DEFAULT_PERSONA`
string and emits exactly one senselog line naming the path it tried and why.
An empty system prompt is treated as worse than no file at all, so an empty
file falls back too rather than shipping a blank persona. This total-fallback
design exists because a wheel install ships `reachy_nova/` only — the
repo-root `config/` tree is not in the package (packaging boundary, spec
c34) — so `NOVA_PERSONA_PATH` pointing at nothing must still leave Nova with
a personality, not a silent, unexplained voice change.

## What is NOT here

Tool mechanics. `app.py`'s `TOOL_GUIDE` — which tool moves what, what to say
when the engine refuses a move, when to call `recall_senses` — is appended
separately by `build_system_prompt`, never written into the persona file.
This keeps the persona a pure description of a character and the tool
contract a pure description of mechanics, so either can be rewritten without
touching the other. The word "assistant" appears in neither half on purpose.

## The embedded fallback

`persona.DEFAULT_PERSONA` is a shorter version of the same character, kept in
the identical register on purpose — a fallback that sounded like a help desk
would turn a missing file into a different robot, defeating the whole reason
this design exists. It is exercised whenever `NOVA_PERSONA_PATH` is set to a
typo, the repo-root `config/` tree is absent (a wheel install), or the file
is otherwise unusable.

## Knobs

| Knob | Where | Default | Notes |
| :--- | :--- | :--- | :--- |
| `NOVA_PERSONA_PATH` | environment | unset (repo path) | Overrides the persona file location. Doubles as the round's kill switch for a persona experiment gone wrong: point it at any file, or leave the field blank to fall back to the embedded default. |
| `SONIC_VOICE_ID` | `harness/app.py` module constant | `"amy"` (Nova 2 Sonic, en-GB) | Not an env var — the voice is a code-level decision (spec: "English-only voice — matthew/tiffany remain the only polyglot options if other languages are ever wanted"). The system prompt steers lexical style, not accent or pitch, so voice choice is the only knob on the *sound* of the personality. |

## Log lines to grep

```text
[SENSE stage=supervise source=nova event=persona] persona file unusable path=<path> reason=<missing|unreadable (...)|undecodable (...)|empty>; using embedded default
[SENSE stage=supervise source=nova event=persona] persona source=<file:<path>|embedded> chars=<n>
```

The first line only fires on a fallback (`persona.py`'s `_fall_back`); the
second is `app.py`'s own startup line and fires every boot regardless of
outcome, so "which persona is this session actually running" is always one
grep away, on every restart, embedded or not.

## Failure modes

- **Missing file** (a mistyped `NOVA_PERSONA_PATH`, or a wheel install with
  no `config/` tree): embedded default, one named line.
- **Unreadable file** (permissions, a broken symlink): embedded default, the
  underlying `OSError` named in the line.
- **A directory at the path**: also an `OSError` on read, same fallback.
- **Undecodable bytes** (a binary file dropped at the persona path): embedded
  default, the `UnicodeDecodeError` reason named.
- **Empty or whitespace-only file**: embedded default — deliberately treated
  as a failure, not "an empty persona," since an empty system prompt is
  worse than none.
- None of these ever raise out of `read()`/`load()`/`source()` — the
  function is total, matching the spec's honesty condition that a wheel
  install "starts the harness and logs the embedded default persona in use
  plus one senselog line naming the missing file."

## The measured facts behind the design

- The live persona before this round measured about 1,500 characters, of
  which roughly two-thirds was tool mechanics and one clause was character —
  the imbalance this round corrects by moving mechanics out entirely.
- AWS's Nova 2 Sonic guide documents that the model over-uses explicit
  phrase lists it is given and recommends one-shot examples instead for
  natural variation — the reason the persona file shows two example
  exchanges rather than a list of lines Nova should or should not say.
- A live 10-minute session is the honesty condition for the persona's actual
  effect: Nova never offering help unasked or asking how it can help, and
  reacting to two pats with different words, is checked live, not by test
  alone (spec c8/h5).
