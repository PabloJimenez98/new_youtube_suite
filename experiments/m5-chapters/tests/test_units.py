from lib.units import Atom, UnitConfig, build_units


def words(*specs):
    return [Atom(s, e, t) for s, e, t in specs]


def test_sentence_mode_closes_at_terminal_punctuation():
    atoms = words((0, 1, "Hello"), (1, 3, "world."), (3, 4, "Next"), (4, 6, "one?"))
    units = build_units(atoms, UnitConfig(mode="sentence", min_s=1))
    assert [(u.start, u.end, u.text) for u in units] == [
        (0, 3, "Hello world."),
        (3, 6, "Next one?"),
    ]


def test_sentence_mode_ignores_punctuation_before_min_s():
    atoms = words((0, 0.5, "Yes."), (0.5, 3, "I agree."))
    units = build_units(atoms, UnitConfig(mode="sentence", min_s=2))
    assert [u.text for u in units] == ["Yes. I agree."]


def test_sentence_mode_caps_at_max_s():
    atoms = [Atom(float(i), float(i) + 1, "w") for i in range(40)]  # no punctuation
    units = build_units(atoms, UnitConfig(mode="sentence", max_s=15))
    assert all(u.end - u.start <= 15 for u in units)
    assert units[0].start == 0 and units[1].start == 15


def test_block_mode_fixed_duration():
    atoms = [Atom(float(i), float(i) + 1, f"w{i}.") for i in range(25)]
    units = build_units(atoms, UnitConfig(mode="block", block_s=10))
    assert [u.start for u in units] == [0, 10, 20]


def test_untimed_atoms_join_the_open_unit_and_never_start_one():
    atoms = [Atom(None, None, "$5"), Atom(0, 1, "costs"), Atom(None, None, "€"), Atom(1, 3, "only.")]
    units = build_units(atoms, UnitConfig(min_s=1))
    assert [(u.start, u.end, u.text) for u in units] == [(0, 3, "$5 costs € only.")]


def test_empty_text_and_no_timed_atoms():
    assert build_units([Atom(0, 1, "  ")]) == []
    assert build_units([Atom(None, None, "x")]) == []


def test_single_huge_segment_atom_is_one_unit():
    units = build_units([Atom(0, 120, "one very long vidchapters sentence")])
    assert [(u.start, u.end) for u in units] == [(0, 120)]
