"""Automated architectural and repo-hygiene guardrails derived from /retro."""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_SRC = REPO_ROOT / "backend" / "src" / "weather_quant"
PYPROJECT_TOML = REPO_ROOT / "backend" / "pyproject.toml"
ENV_EXAMPLE = REPO_ROOT / ".env.example"
GITIGNORE = REPO_ROOT / ".gitignore"
FRONTEND_SRC = REPO_ROOT / "frontend" / "src"


def _package_name(req: str) -> str:
    # Strip extras and version specifiers (e.g. "uvicorn[standard]>=0.30" -> "uvicorn")
    base = re.split(r"[<>=\[!~; ]", req.strip(), maxsplit=1)[0]
    return base.lower().replace("-", "_")


def test_pyproject_declares_all_backend_third_party_imports() -> None:
    """Every third-party module imported in backend/src/weather_quant must appear in pyproject.toml."""
    data = tomllib.loads(PYPROJECT_TOML.read_text(encoding="utf-8"))
    declared = {_package_name(dep) for dep in data["project"]["dependencies"]}
    # Add known module-name aliases if any
    declared_aliases = set(declared)

    missing: list[str] = []
    for py_file in sorted(BACKEND_SRC.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            top: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    _check_third_party(top, py_file, declared_aliases, missing)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                top = node.module.split(".")[0]
                _check_third_party(top, py_file, declared_aliases, missing)

    assert not missing, f"Undeclared third-party imports in pyproject.toml: {sorted(set(missing))}"


def _check_third_party(
    top: str,
    py_file: Path,
    declared: set[str],
    missing: list[str],
) -> None:
    if top in sys.stdlib_module_names or top == "weather_quant":
        return
    if top.lower().replace("-", "_") not in declared:
        missing.append(f"{py_file.relative_to(REPO_ROOT)}:{top}")


def test_no_cross_module_private_imports() -> None:
    """Disallow importing private `_symbol` names from other weather_quant modules."""
    violations: list[str] = []
    for py_file in sorted(BACKEND_SRC.rglob("*.py")):
        rel_parts = py_file.relative_to(BACKEND_SRC).with_suffix("").parts
        if rel_parts[-1] == "__init__":
            rel_parts = rel_parts[:-1]
        current_mod = "weather_quant" + ("." + ".".join(rel_parts) if rel_parts else "")

        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("weather_quant."):
                if node.module == current_mod:
                    continue
                for alias in node.names:
                    if alias.name.startswith("_"):
                        violations.append(
                            f"{py_file.relative_to(REPO_ROOT)} imports {alias.name} from {node.module}"
                        )

    assert not violations, f"Cross-module private imports violate deep-module seams: {violations}"


def test_http_routes_avoid_raw_sql_execution() -> None:
    """Route handlers must call repository/store methods rather than raw SQL."""
    api_dir = BACKEND_SRC / "api"
    offenders: list[str] = []
    for route_file in sorted(api_dir.glob("routes_*.py")):
        text = route_file.read_text(encoding="utf-8")
        if ".conn.execute(" in text or ".executescript(" in text:
            offenders.append(str(route_file.relative_to(REPO_ROOT)))
    assert not offenders, f"Raw SQL found inside API route module(s): {offenders}"


def test_env_example_and_gitignore_parity() -> None:
    """Verify NEXT_PUBLIC_* vars match frontend usage and runtime artifacts stay gitignored."""
    env_lines = ENV_EXAMPLE.read_text(encoding="utf-8").splitlines()
    next_public_vars = [
        line.split("=", 1)[0].strip()
        for line in env_lines
        if line.strip() and not line.strip().startswith("#") and line.strip().startswith("NEXT_PUBLIC_")
    ]
    assert next_public_vars, "Expected at least one NEXT_PUBLIC_* var in .env.example"

    frontend_corpus = "\n".join(
        p.read_text(encoding="utf-8")
        for p in sorted(FRONTEND_SRC.rglob("*"))
        if p.suffix in {".ts", ".tsx"}
    )
    for var_name in next_public_vars:
        assert var_name in frontend_corpus, f"{var_name} in .env.example is unused in frontend/src"

    gitignore_text = GITIGNORE.read_text(encoding="utf-8")
    for required in (".env", "data/**/*.parquet", "data/models/", "logs/", ".venv/", "node_modules/"):
        assert required in gitignore_text, f"Missing required .gitignore pattern: {required}"
