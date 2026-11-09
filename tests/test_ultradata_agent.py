from scripts.prepare_ultradata_agent import split_for_id, valid_trajectory


def test_split_is_deterministic_and_valid():
    assert split_for_id("sample-1") == split_for_id("sample-1")
    assert split_for_id("sample-1") in {"train", "validation", "test"}


def test_valid_trajectory_accepts_basic_english_messages():
    row = {
        "uuid": "sample-1",
        "messages": [
            {"role": "user", "content": "Find the current exchange rate."},
            {"role": "assistant", "content": "I will search for the rate."},
        ],
        "tools": [],
    }
    assert valid_trajectory(row, 1000) == (True, "accepted")


def test_valid_trajectory_rejects_chinese_and_long_rows():
    base = {"uuid": "sample-2", "messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好"}], "tools": []}
    assert valid_trajectory(base, 1000) == (False, "non_english_or_mixed_language")
    long_row = {"uuid": "sample-3", "messages": [{"role": "user", "content": "x" * 200}, {"role": "assistant", "content": "ok"}], "tools": []}
    assert valid_trajectory(long_row, 100) == (False, "trajectory_too_long")
