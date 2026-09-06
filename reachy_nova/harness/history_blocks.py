"""Shape conversation-history blocks the way Bedrock's chat history demands.

LIVE INCIDENT (robot, 2026-09-06 00:12-00:18): the first memory replay after
a restart began with one of Nova's own lines — the ledger's first entry was a
pat reaction, nobody had spoken yet — and Bedrock refused the stream with
``First message in chat history should not be Assistant``. The harness then
replayed the same history on every restart, so the mind sat in a restart loop
at the 60 s backoff cap for six minutes. Two lessons, both here:

* history must START with a USER block and ALTERNATE roles; consecutive blocks
  of one role are merged, a leading ASSISTANT run is dropped, and a trailing
  USER block is dropped when anything else remains (a history that ends on the
  person's last words invites the model to answer them unprompted — c31 says a
  rotation must not produce speech);
* the normaliser is applied BOTH where the history is built (the compactor)
  and where it is sent (``nova_sonic._replay_history``), because the sender
  is the one whose stream dies if a provider ever gets it wrong.

stdlib only; imports nothing from the harness.
"""

from __future__ import annotations

HISTORY_ROLES = ("USER", "ASSISTANT")


def normalise_history(blocks: list[dict]) -> list[dict[str, str]]:
    """Return *blocks* as a USER-first, role-alternating list of ``{role, text}``.

    * roles are upper-cased and anything outside ``USER``/``ASSISTANT`` or with
      blank text is dropped;
    * a leading run of ASSISTANT blocks is dropped (Bedrock: the first message
      must not be the assistant);
    * consecutive blocks with the same role are merged, texts joined by a
      newline, so the roles alternate;
    * a trailing USER block is dropped when at least one other block remains,
      so the history ends on the assistant and asks nothing of the model.
    """
    cleaned: list[dict[str, str]] = []
    for block in blocks or []:
        clean = _clean_block(block)
        if clean is None:
            continue
        _append_merging_roles(cleaned, clean)
    while cleaned and cleaned[0]["role"] != "USER":
        cleaned.pop(0)
    if len(cleaned) > 1 and cleaned[-1]["role"] == "USER":
        cleaned.pop()
    return cleaned


def _clean_block(block: object) -> dict[str, str] | None:
    """``{role, text}`` with the role upper-cased and the text stripped, or ``None``.

    ``None`` for anything that is not a dict, carries a role outside
    ``USER``/``ASSISTANT``, or has blank text — the drop rules above.
    """
    if not isinstance(block, dict):
        return None
    role = str(block.get("role", "") or "").strip().upper()
    text = str(block.get("text", "") or "").strip()
    if role not in HISTORY_ROLES or not text:
        return None
    return {"role": role, "text": text}


def _append_merging_roles(cleaned: list[dict[str, str]], block: dict[str, str]) -> None:
    """Append *block*, or merge it into the tail when the tail has the same role."""
    if cleaned and cleaned[-1]["role"] == block["role"]:
        merged_text = cleaned[-1]["text"] + "\n" + block["text"]
        cleaned[-1] = {"role": block["role"], "text": merged_text}
    else:
        cleaned.append(block)
