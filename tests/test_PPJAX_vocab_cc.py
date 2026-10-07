# Origin: cc (Claude Code, 2026-10-02) - written for PPJAX.
# Purpose: the frozen vocabulary must match what the model was trained against - a silent token
#          mismatch is the one hazard with no weight-shape check to catch it.
import pytest

from ppjax.tokens import get_encoding, vocab_encoding


def test_v6_vocabulary():
    enc = get_encoding("v6")
    assert enc.n_tokens == 30
    assert enc.n_atoms_per_token == 37
    assert enc.tokens[20] == "UNK"
    assert enc.tokens[-9:] == ("HIS-P", "HIS-S", "HIS-A", "ASP-P", "ASP-D", "ASP-A",
                               "GLU-P", "GLU-D", "GLU-A")
    assert "HID" not in enc.token_to_idx and "HIE" not in enc.token_to_idx


def test_canonical_map_folds_microstates_onto_parents():
    enc = get_encoding("v6")
    t2i, cmap = enc.token_to_idx, enc.canonical_map()
    for variant, parent in (("HIS-P", "HIS"), ("HIS-S", "HIS"), ("ASP-D", "ASP"),
                            ("GLU-P", "GLU")):
        assert cmap[t2i[variant]] == t2i[parent]
    assert cmap[t2i["ALA"]] == t2i["ALA"]


def test_backbone_atom_indices_are_shared_by_every_token():
    enc = get_encoding("v6")
    table = enc.atom_indices(("N", "CA", "C", "O"))
    assert table.shape == (30, 4)
    assert (table == table[0]).all()
    assert table[0].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("name,n", [("v6", 30), ("v4", 32), ("v3", 32)])
def test_vocab_name_selection(name, n):
    assert vocab_encoding(name).n_tokens == n


def test_legacy_bool_selects_the_32_token_set():
    assert vocab_encoding(True).n_tokens == 32
    assert vocab_encoding(False).n_tokens == 21
