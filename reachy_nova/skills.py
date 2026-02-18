"""Skills system for Reachy Nova.

Discovers, loads, and manages skills following the Agent Skills pattern.
Skills are folders with SKILL.md files containing YAML frontmatter metadata
and markdown body instructions.
"""

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Default skills directory is reachy_nova/skills/
DEFAULT_SKILLS_DIR = Path(__file__).parent / "skills"


@dataclass
class Skill:
    """A discovered skill with metadata and optional executor."""

    name: str
    description: str
    body: str  # full markdown body from SKILL.md
    metadata: dict = field(default_factory=dict)
    input_schema: dict = field(default_factory=dict)
    executor: Callable[[dict], str] | None = None


def _parse_skill_md(path: Path) -> dict:
    """Parse a SKILL.md file, extracting YAML frontmatter and body."""
    text = path.read_text()

    # Extract YAML frontmatter between --- markers
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", text, re.DOTALL)
    if not fm_match:
        return {"name": path.parent.name, "description": "", "body": text}

    frontmatter_str = fm_match.group(1)
    body = fm_match.group(2).strip()

    # Simple YAML parsing (avoids PyYAML dependency for basic frontmatter)
    meta = {}
    current_key = None
    current_value_lines = []

    for line in frontmatter_str.split("\n"):
        # Check for key: value
        kv_match = re.match(r"^(\w[\w-]*):\s*(.*)", line)
        if kv_match:
            # Save previous key
            if current_key:
                meta[current_key] = "\n".join(current_value_lines).strip()
            current_key = kv_match.group(1)
            current_value_lines = [kv_match.group(2).lstrip(">").strip()]
        elif current_key and line.startswith("  "):
            current_value_lines.append(line.strip())

    if current_key:
        meta[current_key] = "\n".join(current_value_lines).strip()

    return {
        "name": meta.get("name", path.parent.name),
        "description": meta.get("description", ""),
        "body": body,
        "metadata": {k: v for k, v in meta.items() if k not in ("name", "description")},
    }


class SkillManager:
    """Discovers, registers, and executes skills."""

    def __init__(self, skills_dir: Path | None = None):
        self.skills_dir = skills_dir or DEFAULT_SKILLS_DIR
        self.skills: dict[str, Skill] = {}

    def discover(self) -> None:
        """Scan skills_dir for folders containing SKILL.md files."""
        if not self.skills_dir.is_dir():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return

        for skill_md in self.skills_dir.glob("*/SKILL.md"):
            try:
                parsed = _parse_skill_md(skill_md)
                name = parsed["name"]
                self.skills[name] = Skill(
                    name=name,
                    description=parsed["description"],
                    body=parsed["body"],
                    metadata=parsed.get("metadata", {}),
                )
                logger.info(f"Discovered skill: {name}")
            except Exception as e:
                logger.error(f"Error loading skill from {skill_md}: {e}")

    def register_executor(
        self,
        name: str,
        executor: Callable[[dict], str],
        input_schema: dict | None = None,
    ) -> None:
        """Register an executor callable for a skill.

        If the skill was already discovered, attaches the executor.
        Otherwise creates a minimal skill entry.
        """
        if name in self.skills:
            self.skills[name].executor = executor
            if input_schema:
                self.skills[name].input_schema = input_schema
        else:
            self.skills[name] = Skill(
                name=name,
                description="",
                body="",
                executor=executor,
                input_schema=input_schema or {},
            )
        logger.info(f"Registered executor for skill: {name}")

    def get_tool_specs(self) -> list[dict]:
        """Return tool specs formatted for Nova Sonic's toolConfiguration."""
        tools = []
        for skill in self.skills.values():
            if skill.executor is None:
                continue

            schema = skill.input_schema or {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to ask or look for",
                    }
                },
                "required": ["query"],
            }

            tools.append({
                "toolSpec": {
                    "name": skill.name,
                    "description": skill.description,
                    "inputSchema": {"json": json.dumps(schema)},
                }
            })
        return tools

    def execute(self, tool_name: str, params: dict) -> str:
        """Execute a skill by name, return result string."""
        skill = self.skills.get(tool_name)
        if not skill:
            return f"[Unknown skill: {tool_name}]"
        if not skill.executor:
            return f"[Skill '{tool_name}' has no executor]"
        try:
            return skill.executor(params)
        except Exception as e:
            logger.error(f"Skill '{tool_name}' execution error: {e}")
            return f"[Skill error: {e}]"

    def get_system_context(self) -> str:
        """Return available skills description for the system prompt."""
        if not self.skills:
            return ""

        lines = ["You can also do these things:"]
        for skill in self.skills.values():
            if skill.executor:
                lines.append(f"- {skill.name}: {skill.description}")
        return "\n".join(lines)
