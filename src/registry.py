"""
Prompt Registry — YAML-backed prompt versioning system.

Prompts are stored as YAML files in the prompts/ directory.
Each file is a PromptTemplate with a name, version, template string, and metadata.
This is the ground truth: prompts are git-trackable, human-readable, and auditable.
"""
import os
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional

import yaml

from src.config import config


@dataclass
class PromptTemplate:
    name: str
    version: str
    description: str
    task_type: str
    template: str
    variables: List[str]
    author: str
    created_at: str
    tags: Dict[str, str] = field(default_factory=dict)

    def render(self, **kwargs) -> str:
        """Fill template variables. Raises ValueError for missing variables."""
        missing = [v for v in self.variables if v not in kwargs]
        if missing:
            raise ValueError(
                f"Prompt '{self.name}' requires variables: {missing}. "
                f"Provided: {list(kwargs.keys())}"
            )
        result = self.template
        for k, v in kwargs.items():
            result = result.replace("{" + k + "}", str(v))
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "task_type": self.task_type,
            "variables": ", ".join(self.variables),
            "author": self.author,
            "created_at": self.created_at,
            "tags": str(self.tags),
        }


class PromptRegistry:
    """
    Loads and caches prompt templates from YAML files.
    Provides versioning, rendering, and diff comparison.
    """

    def __init__(self, prompts_dir: str = None):
        self.prompts_dir = Path(prompts_dir or config.PROMPTS_DIR)
        self._cache: Dict[str, PromptTemplate] = {}

    def load_all(self) -> Dict[str, PromptTemplate]:
        """Scan prompts_dir for *.yaml files, return dict keyed by template name."""
        if not self.prompts_dir.exists():
            raise FileNotFoundError(
                f"Prompts directory not found: {self.prompts_dir}. "
                "Create it and add .yaml prompt files."
            )
        self._cache = {}
        for path in sorted(self.prompts_dir.glob("*.yaml")):
            pt = self._parse_yaml(path)
            self._cache[pt.name] = pt
        return dict(self._cache)

    def get(self, name: str) -> PromptTemplate:
        """Retrieve prompt by name. Loads from disk if cache is empty."""
        if not self._cache:
            self.load_all()
        if name not in self._cache:
            available = list(self._cache.keys())
            raise KeyError(
                f"Prompt '{name}' not found. Available: {available}"
            )
        return self._cache[name]

    def list_prompts(self) -> List[str]:
        """Return sorted list of available prompt names."""
        if not self._cache:
            self.load_all()
        return sorted(self._cache.keys())

    def render(self, name: str, **kwargs) -> str:
        """Render a prompt template with provided variables."""
        return self.get(name).render(**kwargs)

    def compare_versions(self, name_a: str, name_b: str) -> Dict[str, Any]:
        """
        Return a line diff summary between two prompt templates.
        Shows what changed in the challenger prompt (B) vs. baseline (A).
        """
        pt_a = self.get(name_a)
        pt_b = self.get(name_b)
        lines_a = pt_a.template.splitlines(keepends=True)
        lines_b = pt_b.template.splitlines(keepends=True)

        diff = list(difflib.unified_diff(
            lines_a, lines_b,
            fromfile=f"{name_a} (v{pt_a.version})",
            tofile=f"{name_b} (v{pt_b.version})",
            lineterm="",
        ))

        added = [l.lstrip("+") for l in diff if l.startswith("+") and not l.startswith("+++")]
        removed = [l.lstrip("-") for l in diff if l.startswith("-") and not l.startswith("---")]

        return {
            "prompt_a": name_a,
            "prompt_b": name_b,
            "version_a": pt_a.version,
            "version_b": pt_b.version,
            "added_lines": added,
            "removed_lines": removed,
            "char_delta": len(pt_b.template) - len(pt_a.template),
            "unified_diff": "\n".join(diff),
        }

    def get_all_as_table(self) -> List[List[str]]:
        """Return all prompts formatted for Gradio Dataframe."""
        if not self._cache:
            self.load_all()
        return [
            [
                pt.name,
                pt.version,
                pt.task_type,
                pt.description,
                ", ".join(pt.variables),
            ]
            for pt in sorted(self._cache.values(), key=lambda x: x.name)
        ]

    @staticmethod
    def _parse_yaml(path: Path) -> PromptTemplate:
        """Parse a single YAML file into a PromptTemplate."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return PromptTemplate(
            name=data["name"],
            version=str(data.get("version", "1.0.0")),
            description=data.get("description", ""),
            task_type=data.get("task_type", "generic"),
            template=data["template"],
            variables=data.get("variables", []),
            author=data.get("author", "unknown"),
            created_at=str(data.get("created_at", "")),
            tags=data.get("tags", {}),
        )
