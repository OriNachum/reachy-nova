"""normalise_history: the shape Bedrock's chat history demands (live incident 2026-09-06)."""

from reachy_nova.harness.history_blocks import normalise_history


def test_a_leading_assistant_run_is_dropped():
    blocks = [
        {"role": "ASSISTANT", "text": "Thank you!"},
        {"role": "ASSISTANT", "text": "Lovely."},
        {"role": "USER", "text": "hello"},
        {"role": "ASSISTANT", "text": "hi"},
    ]
    assert normalise_history(blocks) == [
        {"role": "USER", "text": "hello"},
        {"role": "ASSISTANT", "text": "hi"},
    ]


def test_consecutive_same_role_blocks_are_merged():
    blocks = [
        {"role": "USER", "text": "(context)"},
        {"role": "user", "text": "hello"},
        {"role": "ASSISTANT", "text": "hi"},
        {"role": "ASSISTANT", "text": "again"},
    ]
    out = normalise_history(blocks)
    assert [b["role"] for b in out] == ["USER", "ASSISTANT"]
    assert out[0]["text"] == "(context)\nhello"
    assert out[1]["text"] == "hi\nagain"


def test_a_trailing_user_block_is_dropped_when_anything_else_remains():
    blocks = [
        {"role": "USER", "text": "(context)"},
        {"role": "ASSISTANT", "text": "hi"},
        {"role": "USER", "text": "the person's last words"},
    ]
    assert normalise_history(blocks) == [
        {"role": "USER", "text": "(context)"},
        {"role": "ASSISTANT", "text": "hi"},
    ]


def test_a_lone_user_block_survives():
    assert normalise_history([{"role": "USER", "text": "(context)"}]) == [
        {"role": "USER", "text": "(context)"}
    ]


def test_unknown_roles_blank_text_and_junk_are_dropped():
    blocks = [
        {"role": "SYSTEM", "text": "no"},
        {"role": "USER", "text": "   "},
        "junk",
        {"role": "USER", "text": "ok"},
        {"role": "ASSISTANT", "text": ""},
        {"role": "ASSISTANT", "text": "fine"},
    ]
    assert normalise_history(blocks) == [
        {"role": "USER", "text": "ok"},
        {"role": "ASSISTANT", "text": "fine"},
    ]


def test_only_assistant_blocks_yield_nothing():
    assert normalise_history([{"role": "ASSISTANT", "text": "Thank you!"}]) == []
