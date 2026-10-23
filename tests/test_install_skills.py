import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from decoupler._skills import install as install_mod
from decoupler._skills.install import main, resolve_dest, source_dir

###
# Helpers


@pytest.fixture(autouse=True)
def fake_home(tmp_path, monkeypatch):
    """Point HOME at a scratch dir so no test ever touches the real agent config."""

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return home


def _link_targets(text):
    """Return the basenames of every Markdown link to a .md file in `text`."""

    return [m.split("/")[-1] for m in re.findall(r"\]\(([^)]+\.md)\)", text)]


def _snapshot(root):
    """Map each file under `root` to its bytes."""

    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


###
# Bundled-skill integrity (run against the data that actually ships)


def test_bundled_skill_dir_is_well_formed():
    data = source_dir()
    assert data.is_dir()
    assert (data / "SKILL.md").is_file()

    refs = list((data / "references").glob("*.md"))
    assert len(refs) > 0


def test_skill_frontmatter():
    text = (source_dir() / "SKILL.md").read_text()
    assert text.startswith("---\n")

    frontmatter = text.split("---\n", 2)[1]
    fields = dict(re.findall(r"^(\w+):\s*(.*)$", frontmatter, flags=re.MULTILINE))

    assert fields.get("name") == "decoupler"
    # Claude Code caps the description (combined with when_to_use) at 1536 chars.
    assert 0 < len(fields.get("description", "")) <= 1536


def test_internal_links_all_resolve():
    data = source_dir()
    refs = data / "references"
    existing = {p.name for p in refs.glob("*.md")}

    broken = []
    for f in [data / "SKILL.md", *refs.glob("*.md")]:
        for target in _link_targets(f.read_text()):
            if target not in existing:
                broken.append(f"{f.name} -> {target}")

    assert broken == [], f"broken internal links: {broken}"


def test_missing_bundle_raises(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    (empty / "data").mkdir(parents=True)
    monkeypatch.setattr(install_mod, "resources", SimpleNamespace(files=lambda _: empty))

    with pytest.raises(RuntimeError, match="missing"):
        source_dir()
    assert main(["--dest", str(tmp_path / "decoupler")]) == 1


###
# Destination resolution


def test_dest_overrides_agent(tmp_path):
    assert resolve_dest("claude", tmp_path / "x") == tmp_path / "x"


def test_no_agent_or_dest_raises():
    with pytest.raises(ValueError, match="--agent"):
        resolve_dest(None, None)


@pytest.mark.parametrize(
    ("agent", "root"),
    [("claude", ".claude"), ("codex", ".agents"), ("agents", ".agents")],
)
def test_agent_dest_is_under_home(fake_home, agent, root):
    assert resolve_dest(agent, None) == fake_home / root / "skills" / "decoupler"


def test_claude_honours_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    assert resolve_dest("claude", None) == tmp_path / "cfg" / "skills" / "decoupler"


def test_claude_science_uses_active_org(fake_home):
    science = fake_home / ".claude-science"
    science.mkdir()
    (science / "active-org.json").write_text(json.dumps({"org_uuid": "org-1234"}))

    expected = science / "orgs" / "org-1234" / "skills" / "decoupler"
    assert resolve_dest("claude-science", None) == expected


@pytest.mark.parametrize("content", [None, "not json", json.dumps({"org_uuid": "../escape"})])
def test_claude_science_bad_active_org_raises(fake_home, content):
    science = fake_home / ".claude-science"
    science.mkdir()
    if content is not None:
        (science / "active-org.json").write_text(content)

    with pytest.raises(ValueError, match="--dest"):
        resolve_dest("claude-science", None)


###
# main() behavior (always into a temporary destination)


def test_install_into_fresh_dest(tmp_path):
    dest = tmp_path / "decoupler"
    assert main(["--dest", str(dest)]) == 0
    assert _snapshot(dest) == _snapshot(source_dir())


def test_install_via_agent(fake_home):
    assert main(["--agent", "claude"]) == 0
    assert (fake_home / ".claude" / "skills" / "decoupler" / "SKILL.md").is_file()


def test_install_without_target_fails(capsys):
    assert main([]) == 1
    assert "--agent" in capsys.readouterr().err


def test_reinstall_identical_is_a_noop(tmp_path, capsys):
    dest = tmp_path / "decoupler"
    main(["--dest", str(dest)])

    assert main(["--dest", str(dest)]) == 0
    assert "already up to date" in capsys.readouterr().out


def test_install_differing_dest_fails_without_force(tmp_path):
    dest = tmp_path / "decoupler"
    main(["--dest", str(dest)])
    junk = dest / "references" / "stale.md"
    junk.write_text("keep me")

    assert main(["--dest", str(dest)]) == 1
    assert junk.exists()


def test_install_force_replaces_stale_content(tmp_path):
    dest = tmp_path / "decoupler"
    main(["--dest", str(dest)])
    junk = dest / "references" / "stale.md"
    junk.write_text("remove me")

    assert main(["--dest", str(dest), "--force"]) == 0
    assert not junk.exists()
    assert _snapshot(dest) == _snapshot(source_dir())


def test_check_reports_install_state(tmp_path):
    dest = tmp_path / "decoupler"
    assert main(["--dest", str(dest), "--check"]) == 1
    assert not dest.exists()

    main(["--dest", str(dest)])
    assert main(["--dest", str(dest), "--check"]) == 0

    (dest / "SKILL.md").write_text("edited")
    assert main(["--dest", str(dest), "--check"]) == 1


def test_print_path(capsys):
    assert main(["--print-path"]) == 0
    assert Path(capsys.readouterr().out.strip()) == source_dir()


###
# Console-script entry point (opt-in; does not install anything)


def test_console_script_print_path():
    # The entry point is installed next to the running interpreter (bin/Scripts).
    exe = Path(sys.executable).parent / "decoupler-install-skills"
    if not exe.exists():
        pytest.skip("decoupler-install-skills entry point is not installed")

    result = subprocess.run([str(exe), "--print-path"], capture_output=True, text=True)

    assert result.returncode == 0
    assert result.stdout.strip().endswith("data")
