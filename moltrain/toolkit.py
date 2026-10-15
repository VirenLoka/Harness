"""The tool surface the policy is trained against.

These are the same functions the MCP server exposes: `server.py` registers
them with FastMCP, and this module imports and calls them directly. There is
no MCP round trip and no Node runtime, so training runs in a bare container,
while the tools the model learns stay byte-identical to the deployed ones.

Two composites that live in the TypeScript harness plugin are reimplemented
here in Python so the trained tool surface matches what the agent sees in the
harness.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import logs
from .config import resolve

LOG = logs.get_logger("toolkit")


@dataclass
class ToolResult:
    """One tool invocation, successful or not. Never raises at the call site."""

    name: str
    ok: bool
    content: Any
    error: str | None = None
    duration_s: float = 0.0
    cached: bool = False

    def as_text(self) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        if isinstance(self.content, str):
            return self.content
        return json.dumps(self.content, indent=2, sort_keys=False, default=str)


@dataclass
class ToolSpec:
    name: str
    fn: Callable[..., Any]
    is_async: bool
    description: str
    parameters: dict[str, Any]
    terminal: bool = False  # ends the episode when called


@dataclass
class CallLog:
    """Timing and outcome of every call in one episode, for cost analysis."""

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, result: ToolResult) -> None:
        self.entries.append(
            {
                "tool": result.name,
                "ok": result.ok,
                "duration_s": round(result.duration_s, 3),
                "cached": result.cached,
            }
        )

    @property
    def total_seconds(self) -> float:
        return sum(entry["duration_s"] for entry in self.entries)

    def seconds_by_tool(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for entry in self.entries:
            totals[entry["tool"]] = totals.get(entry["tool"], 0.0) + entry["duration_s"]
        return totals


def import_molbench(cfg: dict[str, Any]):
    """Import molbench's server module and point its output dirs at this run."""
    molbench_dir = resolve(cfg, cfg["molbench"]["path"])
    if not (molbench_dir / "server.py").is_file():
        raise FileNotFoundError(f"molbench server.py not found under {molbench_dir}")

    import os

    for key, env in (
        ("receptor_dir", "MOLBENCH_RECEPTOR_DIR"),
        ("docking_dir", "MOLBENCH_DOCKING_DIR"),
        ("viz_dir", "MOLBENCH_VIZ_DIR"),
    ):
        value = cfg["molbench"].get(key)
        if value:
            os.environ[env] = str(resolve(cfg, value))

    if str(molbench_dir) not in sys.path:
        sys.path.insert(0, str(molbench_dir))
    import server

    return server


def _summarise_arguments(arguments: dict[str, Any], limit: int = 120) -> str:
    """Arguments on one line, with long values clipped.

    A receptor path or a reaction SMARTS can be long enough to drown the log,
    and the point of this line is to see which tool ran on what.
    """
    parts = []
    for key, value in arguments.items():
        text = str(value)
        if len(text) > limit:
            text = text[:limit] + "..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def _describe(fn: Callable[..., Any]) -> tuple[str, dict[str, Any]]:
    """Docstring summary and parameter names/annotations for one tool."""
    doc = inspect.getdoc(fn) or ""
    signature = inspect.signature(fn)
    parameters = {
        name: {
            "required": parameter.default is inspect.Parameter.empty,
            "annotation": str(parameter.annotation),
        }
        for name, parameter in signature.parameters.items()
    }
    return doc, parameters


class Toolkit:
    """Resolves tool names to callables and executes them safely."""

    def __init__(self, cfg: dict[str, Any], server_module: Any):
        self.cfg = cfg
        self.server = server_module
        self.timeout_s = float(cfg["episode"]["tool_timeout_s"])
        self.specs: dict[str, ToolSpec] = {}
        self._register_builtin()

    def _register_builtin(self) -> None:
        enabled = list(self.cfg["episode"]["tools"])
        for name in enabled:
            fn = getattr(self.server, name, None)
            if fn is None:
                fn = getattr(self, f"_composite_{name}", None)
                if fn is None:
                    raise ValueError(f"unknown tool in episode.tools: {name}")
            doc, parameters = _describe(fn)
            self.specs[name] = ToolSpec(
                name=name,
                fn=fn,
                is_async=inspect.iscoroutinefunction(fn),
                description=doc,
                parameters=parameters,
            )
        # Always present: the policy ends an episode by submitting a molecule,
        # which is far more reliable than parsing a SMILES out of prose.
        doc, parameters = _describe(self._submit_molecule)
        self.specs["submit_molecule"] = ToolSpec(
            name="submit_molecule",
            fn=self._submit_molecule,
            is_async=False,
            description=doc,
            parameters=parameters,
            terminal=True,
        )

    # ------------------------------------------------------------ tool bodies

    def _submit_molecule(self, smiles: str, rationale: str = "") -> dict[str, Any]:
        """Submit the final optimized molecule and end the task.

        Args:
            smiles: SMILES of the molecule you are proposing as the answer.
            rationale: One or two sentences on why this molecule is better.
        """
        return {"submitted_smiles": smiles, "rationale": rationale}

    async def _composite_dock_and_score(
        self,
        smiles: str,
        receptor_file: str,
        engine: str = "vina",
        exhaustiveness: int = 8,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Dock a molecule and report 3D and 2D metrics together.

        Mirrors the harness plugin's composite: validate, dock, then combine
        the docking score with ligand-efficiency and 2D properties.

        Args:
            smiles: SMILES of the ligand to dock.
            receptor_file: Prepared receptor .pdbqt from prepare_target_receptor.
            engine: Docking engine, vina or smina.
            exhaustiveness: Search effort; higher is slower and more thorough.
            label: Filename prefix for the saved pose files.
        """
        validation = self.server.validate_smiles(smiles)
        if not validation["valid"]:
            return {
                "status": "invalid_input",
                "detail": "; ".join(validation["errors"]),
            }
        docked = await self.server.run_docking_simulation(
            smiles=smiles,
            receptor_file=receptor_file,
            engine=engine,
            exhaustiveness=exhaustiveness,
            label=label or "moltrain",
        )
        affinity = docked["binding_affinity"]
        metrics = self.server.get_sbdd_metrics(smiles=smiles, docking_score=affinity)
        properties = self.server.get_molecule_metrics(smiles=smiles)
        return {
            "status": "scored",
            "smiles": properties["canonical_smiles"],
            "engine": docked["engine"],
            "binding_affinity": affinity,
            "ligand_efficiency": metrics["Ligand_Efficiency"],
            "lipophilic_ligand_efficiency": metrics["Lipophilic_Ligand_Efficiency"],
            "estimated_pic50": metrics["estimated_pIC50"],
            "clogp": metrics["cLogP"],
            "tpsa": metrics["TPSA"],
            "sa_score": properties["sa_score"],
            "qed": properties["qed"],
            "exact_mw": properties["exact_mw"],
            "heavy_atoms": metrics["heavy_atoms"],
            "pose_file": docked["pose_file"],
            "viewer_file": docked["viewer_file"],
        }

    async def _composite_benchmark_transformation(
        self,
        smiles: str,
        reaction_smarts: str,
        max_products: int = 3,
    ) -> dict[str, Any]:
        """Apply a reaction SMARTS and score every product against the input.

        Args:
            smiles: SMILES of the starting material.
            reaction_smarts: Single-reactant reaction SMARTS, e.g. '[cH:1]>>[c:1]F'.
            max_products: How many products to score.
        """
        validation = self.server.validate_smiles(smiles)
        if not validation["valid"]:
            return {
                "status": "invalid_input",
                "detail": "; ".join(validation["errors"]),
            }
        start = self.server.get_molecule_metrics(smiles=smiles)
        reaction = self.server.apply_smarts_reaction(
            smiles=smiles, reaction_smarts=reaction_smarts
        )
        scored = []
        for product in reaction["products"][:max_products]:
            metrics = self.server.get_molecule_metrics(smiles=product)
            scored.append(
                {
                    "smiles": metrics["canonical_smiles"],
                    "sa_score": metrics["sa_score"],
                    "qed": metrics["qed"],
                    "exact_mw": metrics["exact_mw"],
                    "sa_delta": round(metrics["sa_score"] - start["sa_score"], 4),
                    "qed_delta": round(metrics["qed"] - start["qed"], 4),
                    "mw_delta": round(metrics["exact_mw"] - start["exact_mw"], 4),
                }
            )
        if not scored:
            return {
                "status": "no_products",
                "detail": f"{reaction_smarts} produced no valid products",
                "reactant": start,
            }
        scored.sort(key=lambda row: row["sa_delta"])
        return {
            "status": "scored",
            "reactant": start,
            "products": scored,
            "num_products": reaction["num_products"],
        }

    # ------------------------------------------------------------- dispatching

    def names(self) -> list[str]:
        return list(self.specs)

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Run one tool call, converting every failure into a value."""
        spec = self.specs.get(name)
        if spec is None:
            known = ", ".join(sorted(self.specs))
            return ToolResult(
                name=name,
                ok=False,
                content=None,
                error=f"unknown tool {name!r}. Available tools: {known}",
            )

        started = time.monotonic()
        LOG.debug("-> %s(%s)", name, _summarise_arguments(arguments))
        try:
            if spec.is_async:
                content = await asyncio.wait_for(
                    spec.fn(**arguments), timeout=self.timeout_s
                )
            else:
                content = await asyncio.wait_for(
                    asyncio.to_thread(spec.fn, **arguments), timeout=self.timeout_s
                )
        except TypeError as exc:  # wrong or missing arguments
            LOG.debug("<- %s rejected its arguments: %s", name, exc)
            return ToolResult(
                name=name,
                ok=False,
                content=None,
                error=f"invalid arguments for {name}: {exc}",
                duration_s=time.monotonic() - started,
            )
        except TimeoutError:
            LOG.warning("%s timed out after %.0fs", name, self.timeout_s)
            return ToolResult(
                name=name,
                ok=False,
                content=None,
                error=f"{name} timed out after {self.timeout_s:.0f}s",
                duration_s=time.monotonic() - started,
            )
        except Exception as exc:  # noqa: BLE001 - a tool failure is data, not a crash
            # ToolError carries the message the MCP client would show, so the
            # policy sees the same wording in training as in the harness. For
            # anything else the exception type is the diagnostic.
            message = (
                str(exc)
                if type(exc).__name__ == "ToolError"
                else f"{type(exc).__name__}: {exc}"
            )
            LOG.debug("<- %s failed: %s", name, message)
            return ToolResult(
                name=name,
                ok=False,
                content=None,
                error=message,
                duration_s=time.monotonic() - started,
            )
        duration = time.monotonic() - started
        LOG.debug("<- %s ok in %.2fs", name, duration)
        return ToolResult(name=name, ok=True, content=content, duration_s=duration)


def build_toolkit(cfg: dict[str, Any]) -> Toolkit:
    return Toolkit(cfg, import_molbench(cfg))


def tool_catalog(toolkit: Toolkit) -> list[dict[str, Any]]:
    """OpenAI-style tool definitions, for rendering the system prompt."""
    catalog = []
    for spec in toolkit.specs.values():
        properties = {
            name: {"type": "string", "description": meta["annotation"]}
            for name, meta in spec.parameters.items()
            if name != "self"
        }
        required = [
            name
            for name, meta in spec.parameters.items()
            if meta["required"] and name != "self"
        ]
        catalog.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description.split("\n\n")[0],
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return catalog
