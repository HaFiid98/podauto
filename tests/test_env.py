import os

from podauto.env import load_env_file


def write(tmp_path, text):
    p = tmp_path / ".env"
    p.write_text(text)
    return p


def test_values_reach_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("PODAUTO_T1", raising=False)
    loaded = load_env_file(write(tmp_path, "PODAUTO_T1=abc123\n"))
    assert loaded == ["PODAUTO_T1"]
    assert os.environ["PODAUTO_T1"] == "abc123"


def test_an_exported_variable_wins(tmp_path, monkeypatch):
    """Shell exports and CI secrets are more deliberate than a file; silently
    overriding them would make the override invisible."""
    monkeypatch.setenv("PODAUTO_T2", "from-shell")
    assert load_env_file(write(tmp_path, "PODAUTO_T2=from-file\n")) == []
    assert os.environ["PODAUTO_T2"] == "from-shell"


def test_comments_blanks_and_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("PODAUTO_T3", raising=False)
    load_env_file(write(tmp_path, '\n# a comment\n  PODAUTO_T3 = "quoted value"  \n'))
    assert os.environ["PODAUTO_T3"] == "quoted value"


def test_a_value_containing_equals_is_kept_whole(tmp_path, monkeypatch):
    """Base64 and JWT-ish credentials end in '=' padding, so the split has to be
    on the first separator only."""
    monkeypatch.delenv("PODAUTO_T4", raising=False)
    load_env_file(write(tmp_path, "PODAUTO_T4=a=b=c==\n"))
    assert os.environ["PODAUTO_T4"] == "a=b=c=="


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == []


def test_returns_names_only(tmp_path, monkeypatch):
    """The caller's natural next move is to log the result, so the return value
    must never carry the values."""
    monkeypatch.delenv("PODAUTO_T5", raising=False)
    loaded = load_env_file(write(tmp_path, "PODAUTO_T5=super-secret\n"))
    assert loaded == ["PODAUTO_T5"]
    assert "super-secret" not in str(loaded)
