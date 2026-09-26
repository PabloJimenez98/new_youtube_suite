"""Unit tests for word/segment speaker assignment -- no model, no DB."""

from app.diarization.assign import (
    SpeakerAssigner,
    Turn,
    relabel_by_first_appearance,
    segment_speaker,
)

TURNS = [
    Turn(0.0, 5.0, "A"),
    Turn(5.0, 9.0, "B"),
    # 9.0-10.0 is a gap (silence, no speaker)
    Turn(10.0, 15.0, "A"),
]


def test_word_inside_a_turn() -> None:
    assigner = SpeakerAssigner(TURNS)
    assert assigner.speaker_for(1.0, 1.4) == "A"
    assert assigner.speaker_for(6.0, 6.5) == "B"
    assert assigner.speaker_for(12.0, 12.2) == "A"


def test_word_straddling_two_turns_takes_the_larger_overlap() -> None:
    assigner = SpeakerAssigner(TURNS)
    assert assigner.speaker_for(4.9, 5.3) == "B"  # 0.1 in A, 0.3 in B
    assert assigner.speaker_for(4.6, 5.1) == "A"  # 0.4 in A, 0.1 in B


def test_word_in_a_gap_takes_the_nearest_turn() -> None:
    assigner = SpeakerAssigner(TURNS)
    assert assigner.speaker_for(9.1, 9.3) == "B"  # midpoint 9.2: 0.2 from B, 0.8 from A
    assert assigner.speaker_for(9.7, 9.9) == "A"  # midpoint 9.8: 0.8 from B, 0.2 from A


def test_word_before_first_or_after_last_turn() -> None:
    assigner = SpeakerAssigner([Turn(2.0, 3.0, "A"), Turn(3.0, 4.0, "B")])
    assert assigner.speaker_for(0.5, 0.8) == "A"
    assert assigner.speaker_for(7.0, 7.5) == "B"


def test_zero_length_word_inside_a_turn() -> None:
    assert SpeakerAssigner(TURNS).speaker_for(6.0, 6.0) == "B"


def test_word_without_timestamps_is_unlabelled() -> None:
    assigner = SpeakerAssigner(TURNS)
    assert assigner.speaker_for(None, None) is None
    assert assigner.speaker_for(1.0, None) is None


def test_no_turns_means_no_labels() -> None:
    assert SpeakerAssigner([]).speaker_for(1.0, 2.0) is None


def test_unsorted_turns_are_handled() -> None:
    assigner = SpeakerAssigner(list(reversed(TURNS)))
    assert assigner.speaker_for(6.0, 6.5) == "B"
    assert assigner.speaker_for(12.0, 12.2) == "A"


def test_segment_speaker_is_the_majority() -> None:
    assert segment_speaker(["B", "A", "A", None, "A"]) == "A"


def test_segment_speaker_tie_goes_to_the_first_tied_speaker() -> None:
    assert segment_speaker([None, "B", "A", "A", "B"]) == "B"


def test_segment_speaker_all_unlabelled() -> None:
    assert segment_speaker([None, None]) is None
    assert segment_speaker([]) is None


def test_relabel_by_first_appearance() -> None:
    raw = [
        Turn(10.0, 12.0, "SPEAKER_00"),
        Turn(0.0, 4.0, "SPEAKER_02"),
        Turn(4.0, 10.0, "SPEAKER_00"),
        Turn(12.0, 13.0, "SPEAKER_01"),
    ]
    assert relabel_by_first_appearance(raw) == [
        Turn(0.0, 4.0, "SPEAKER_00"),
        Turn(4.0, 10.0, "SPEAKER_01"),
        Turn(10.0, 12.0, "SPEAKER_01"),
        Turn(12.0, 13.0, "SPEAKER_02"),
    ]
