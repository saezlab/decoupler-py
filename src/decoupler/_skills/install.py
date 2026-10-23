"""Install the bundled ``decoupler`` skill into an agentic framework's skills directory.

Console script (wired as ``<dist>-install-skills`` in ``pyproject.toml``): copy the skill that
ships inside this package into a chosen framework's skills directory, so an agent can load it.
The same ``SKILL.md`` + ``references/`` bundle is a cross-framework standard, so it installs
verbatim — no per-framework conversion.

Frameworks (``--agent``):

- ``claude``          -> ``~/.claude/skills`` (honours ``CLAUDE_CONFIG_DIR``)
- ``codex``           -> ``~/.agents/skills`` (where Codex discovers user skills; same as ``agents``)
- ``agents``          -> ``~/.agents/skills``
- ``claude-science``  -> the active org's skills dir, resolved from
  ``~/.claude-science/active-org.json``

``--dest`` overrides all of them. There is **no default framework**: pass ``--agent`` or
``--dest``. The skill files live in the ``data/`` directory next to this module and are read via
``importlib.resources``, so this works from an installed wheel, not just an editable checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from importlib import resources
from pathlib import Path

#: The skill name; it installs to ``<framework-skills-dir>/<SKILL_NAME>/``.
SKILL_NAME = "decoupler"

#: Framework -> (env var that overrides the config root, default config root). The skills
#: directory is ``<root>/skills``. ``claude-science`` is resolved separately.
_AGENT_ROOTS = {
    "codex": (None, "~/.agents"),
    "claude": ("CLAUDE_CONFIG_DIR", "~/.claude"),
    "agents": (None, "~/.agents"),
}

#: The frameworks ``--agent`` accepts.
AGENTS = (*sorted(_AGENT_ROOTS), "claude-science")


def source_dir() -> Path:
    """Return the package-owned skill directory (the bundle that gets copied)."""
    source = Path(str(resources.files(__package__).joinpath("data")))
    if not (source / "SKILL.md").is_file():
        raise RuntimeError(f"packaged skill data is missing: {source}")
    return source


def _claude_science_skills_dir() -> Path:
    """Resolve the active Claude Science org's skills directory."""
    root = Path("~/.claude-science").expanduser()
    active_org_path = root / "active-org.json"
    try:
        active_org = json.loads(active_org_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "cannot resolve the Claude Science active organization from "
            f"{active_org_path}; pass --dest instead"
        ) from error
    org_uuid = active_org.get("org_uuid") if isinstance(active_org, dict) else None
    if (
        not isinstance(org_uuid, str)
        or not org_uuid
        or Path(org_uuid).name != org_uuid
        or org_uuid in {".", ".."}
    ):
        raise ValueError(f"invalid Claude Science org_uuid in {active_org_path}; pass --dest instead")
    return root / "orgs" / org_uuid / "skills"


def _skills_dir(agent: str) -> Path:
    """Return the skills directory (parent of the install dir) for a framework."""
    if agent == "claude-science":
        return _claude_science_skills_dir()
    variable, fallback = _AGENT_ROOTS[agent]
    root = os.environ.get(variable) if variable is not None else None
    return Path(root or fallback).expanduser() / "skills"


def resolve_dest(agent: str | None, dest: Path | None) -> Path:
    """Resolve the install destination from ``--agent`` / ``--dest``.

    ``--dest`` wins. With neither, raise ``ValueError`` — there is no default framework.
    """
    if dest is not None:
        return dest.expanduser()
    if agent is None:
        raise ValueError("pass --agent {" + ",".join(AGENTS) + "} or --dest to choose where to install")
    return _skills_dir(agent) / SKILL_NAME


def _snapshot(root: Path) -> dict[str, bytes]:
    """Map each file under ``root`` to its bytes, for exact tree comparison."""
    return {
        str(item.relative_to(root)): item.read_bytes()
        for item in root.rglob("*")
        if item.is_file()
    }


def _matches(source: Path, target: Path) -> bool:
    """Report whether ``target`` is a byte-for-byte copy of ``source``."""
    return target.is_dir() and _snapshot(source) == _snapshot(target)


def main(argv: list[str] | None = None) -> int:
    """Install the bundled skill; entry point for the ``<dist>-install-skills`` script."""
    parser = argparse.ArgumentParser(
        description=f"Install the {SKILL_NAME} skill bundled with this package into an agent's skills directory.",
    )
    parser.add_argument(
        "--agent",
        choices=AGENTS,
        default=None,
        help="framework to install into (no default — pass this or --dest)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="exact skills directory to install into (overrides --agent)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing installation that differs from the bundled skill",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="report whether the installed skill matches the bundled one; do not install",
    )
    action.add_argument(
        "--print-path",
        action="store_true",
        help="print the bundled skill's location inside the package and exit",
    )
    args = parser.parse_args(argv)

    try:
        source = source_dir()
        if args.print_path:
            print(source)
            return 0
        if args.check:
            target = resolve_dest(args.agent, args.dest)
            if not target.exists():
                print(f"{SKILL_NAME} skill is not installed at {target}", file=sys.stderr)
                return 1
            if _matches(source, target):
                print(f"{SKILL_NAME} skill at {target} matches the bundled copy")
                return 0
            print(f"{SKILL_NAME} skill at {target} differs from the bundled copy", file=sys.stderr)
            return 1

        dest = resolve_dest(args.agent, args.dest)
        if dest.exists():
            if _matches(source, dest):
                print(f"{SKILL_NAME} skill already up to date at {dest}")
                return 0
            if not args.force:
                print(f"{dest} already exists and differs; pass --force to overwrite", file=sys.stderr)
                return 1
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"installed the {SKILL_NAME} skill to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
