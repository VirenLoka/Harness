"""The tool surface the policy is trained against.

These are the same functions the MCP server exposes: `server.py` registers
them with FastMCP, and this module imports and calls them directly. There is
no MCP round trip and no Node runtime, so training runs in a bare container,
while the tools the model learns stay byte-identical to the deployed ones.

Two things differ from `moltrain.toolkit`, and both are deliberate.

**The docking tools are gone.** Not disabled, not configurable -- absent. A
PROTAC is decided by the ternary complex, and docking one ligand into one
rigid receptor does not model it; at the ~26 rotatable bonds these molecules
carry the conformational search is past where it stays reliable, and the
score would be dominated by the warhead, which the policy is forbidden to
touch and whose affinity is already known. `protacbench/OPTIMISATION.md`
argues this at length and declines to put docking on the linker tool list.

**`check_protac_linker` is bound to the task's fragments.** The server
signature takes the two fragments as arguments, which on an unbound tool
would let the policy choose what counts as "intact" -- pass a one-ring
substructure of the real warhead and a molecule that deleted most of it
reports `fragments_intact: true`. That is not a bug the reward could catch
either, because `grade.measure` and this tool would then be measuring
different molecules. So the toolkit binds the fragments the task was set
with, and anything the policy passes for them is ignored and said so.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from moltrain.logs import get_logger

from .config import resolve_sibling
from .rewards import _distance

LOG = get_logger("toolkit")

# `moltrain.toolkit` holds the Python reimplementation of the harness's
# TypeScript `benchmark_transformation` composite. It is imported rather than
# copied so the reaction-and-measure half stays one implementation; only the
# ranking is ours, because ranking generated linkers by SA does not work (see
# `_composite_benchmark_transformation`). A missing moltrain is survivable --
# the fallback below does the same work -- so the import is not fatal here.
try:  # pragma: no cover - exercised only by the import failing
    from moltrain.toolkit import Toolkit as _MoltrainToolkit
except Exception:  # noqa: BLE001 - an absent sibling package is not an error
    _MoltrainToolkit = None  # type: ignore[assignment]


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
    """Import molbench's server module, which carries the tool bodies.

    moltrain also exports MOLBENCH_RECEPTOR_DIR and MOLBENCH_DOCKING_DIR here.
    Neither is set: no tool on this surface docks anything, and the only tool
    that writes a file is `render_molecule_2d`, which molbench puts under its
    own `visualizations/` directory.
    """
    molbench_dir = resolve_sibling(cfg, "molbench")
    if not (molbench_dir / "server.py").is_file():
        raise FileNotFoundError(f"molbench server.py not found under {molbench_dir}")

    if str(molbench_dir) not in sys.path:
        sys.path.insert(0, str(molbench_dir))
    import server

    return server


def _summarise_arguments(arguments: dict[str, Any], limit: int = 120) -> str:
    """Arguments on one line, with long values clipped.

    A PROTAC SMILES is ~150 characters and a reaction SMARTS can be longer,
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


def _unwrap(attribute: Any) -> Any:
    """The plain function behind a FastMCP tool object.

    molbench's tools are callable as written -- its own tests call
    `server.check_protac_linker(...)` directly -- but which object `@mcp.tool`
    returns is a FastMCP version detail, and a FunctionTool keeps the original
    on `.fn`. Unwrapping costs one line and turns a future upgrade from a
    confusing TypeError at rollout time into nothing at all.
    """
    return getattr(attribute, "fn", attribute)


class Toolkit:
    """Resolves tool names to callables and executes them safely.

    One instance belongs to one task, because `check_protac_linker` is bound
    to that task's fragments. `for_fragments` makes a sibling that shares the
    imported server module, so binding a new task costs nothing.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        server_module: Any,
        fragment_smiles: Sequence[str] | None = None,
    ):
        self.cfg = cfg
        self.server = server_module
        self.timeout_s = float(cfg["episode"]["tool_timeout_s"])
        self.fragments: list[str] = list(fragment_smiles or [])
        self.thresholds: dict[str, dict[str, Any]] = dict(
            cfg["reward"].get("thresholds") or {}
        )
        self.specs: dict[str, ToolSpec] = {}
        self._register_builtin()

    def for_fragments(self, fragment_smiles: Sequence[str]) -> Toolkit:
        """A toolkit for another task, reusing this one's imported server."""
        return Toolkit(self.cfg, self.server, fragment_smiles)

    def _register_builtin(self) -> None:
        enabled = list(self.cfg["episode"]["tools"])
        for name in enabled:
            # A bound wrapper wins over the raw server function: that is how
            # the task's fragments get frozen into check_protac_linker.
            fn = getattr(self, f"_bound_{name}", None)
            if fn is None:
                fn = getattr(self, f"_composite_{name}", None)
            if fn is None:
                attribute = getattr(self.server, name, None)
                fn = None if attribute is None else _unwrap(attribute)
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
        """Submit the final optimized PROTAC and end the task.

        Args:
            smiles: SMILES of the whole PROTAC you are proposing as the answer,
                with both fragments intact and only the linker changed.
            rationale: One or two sentences on what you changed and why.
        """
        return {"submitted_smiles": smiles, "rationale": rationale}

    def _bound_check_protac_linker(
        self,
        smiles: str,
        fragment_1: str | None = None,
        fragment_2: str | None = None,
    ) -> dict[str, Any]:
        """Check this task's two fragments survived and the linker still bridges them.

        Reports whether both fragments of THIS task are present and unaltered,
        whether the linker is bonded to them exactly twice, how many bonds span
        the two fragments, and whether the linker carries a reactive group.
        `fragments_intact` is the one that matters: a PROTAC missing a fragment
        parses, sanitises, and has excellent descriptors, and will not degrade
        anything.

        Args:
            smiles: SMILES of the whole PROTAC to check.
            fragment_1: Ignored. The fragments are the ones this task was set
                with and cannot be chosen per call.
            fragment_2: Ignored, as above.
        """
        if len(self.fragments) != 2:
            return {
                "status": "unavailable",
                "detail": (
                    "this toolkit was not bound to a fragment pair, so there is "
                    "nothing to check the molecule against"
                ),
            }
        result = dict(
            self.server.check_protac_linker(
                smiles, self.fragments[0], self.fragments[1]
            )
        )
        if fragment_1 is not None or fragment_2 is not None:
            # Said out loud rather than silently dropped: a policy that thinks
            # it narrowed the check and was not told would read a pass as
            # evidence for a claim the tool never made.
            result["note"] = (
                "the fragments you passed were ignored; this check always uses "
                "the two fragments the task was set with"
            )
        return result

    def _band_distance(self, smiles: str) -> float | None:
        """How far outside its target bands a molecule sits, in tolerance units.

        `rewards._distance` is imported rather than re-derived so the tool
        ranks candidates on exactly the arithmetic the reward pays for. Each
        term is divided by its own `tolerance` -- the slack `grade.py` allows
        before the hard limit -- because a cLogP overshoot of 1.0 and a TPSA
        overshoot of 1.0 are not the same size of problem.
        """
        from .properties import measure

        if len(self.fragments) != 2:
            return None
        props = measure(smiles, self.fragments)
        if not props.measurable or not props.fragments_intact:
            return None  # sorts last: a product that ate a fragment is not a candidate

        total = 0.0
        counted = 0
        for name, spec in self.thresholds.items():
            if not spec.get("enabled", True):
                continue
            value = getattr(props, name, None)
            if value is None:
                continue
            tolerance = float(spec.get("tolerance") or 0.0)
            # A band with no stated tolerance is scored in its own raw units;
            # every band in configs/default.yaml states one.
            total += _distance(float(value), spec) / (tolerance if tolerance else 1.0)
            counted += 1
        return round(total, 4) if counted else None

    async def _composite_benchmark_transformation(
        self,
        smiles: str,
        reaction_smarts: str,
        max_products: int = 3,
    ) -> dict[str, Any]:
        """Apply a reaction SMARTS to the linker and score every product.

        Products are ranked by how much closer they bring the molecule to the
        target bands, best first, and each carries whether both fragments
        survived the transformation.

        Args:
            smiles: SMILES of the starting PROTAC.
            reaction_smarts: Single-reactant reaction SMARTS, e.g. '[cH:1]>>[c:1]F'.
            max_products: How many products to score.
        """
        if _MoltrainToolkit is not None:
            # Unbound call: the method reads only `self.server`, which this
            # class has, so the reaction-and-measure half is literally the
            # same code moltrain trains against.
            base = await _MoltrainToolkit._composite_benchmark_transformation(
                self, smiles=smiles, reaction_smarts=reaction_smarts,
                max_products=max_products,
            )
        else:
            base = await self._fallback_benchmark_transformation(
                smiles, reaction_smarts, max_products
            )
        if base.get("status") != "scored":
            return base

        start_distance = self._band_distance(smiles)
        products = []
        for product in base["products"]:
            row = dict(product)
            distance = self._band_distance(row["smiles"])
            row["band_distance"] = distance
            row["fragments_intact"] = distance is not None
            row["band_distance_delta"] = (
                None
                if distance is None or start_distance is None
                else round(distance - start_distance, 4)
            )
            products.append(row)

        # Rank by band distance, NOT by SA. SA is near-flat across generated
        # linkers -- a PEG3 and a C18 alkyl both score about 2 -- which is the
        # reason protacbench/score.py weights motif familiarity alongside it
        # rather than trusting SA to separate candidates. Ranking on SA here
        # would order the products by noise. Products whose fragments did not
        # survive have no band distance and sort to the bottom.
        products.sort(
            key=lambda row: (
                row["band_distance"] is None,
                row["band_distance"] if row["band_distance"] is not None else 0.0,
                row.get("sa_delta", 0.0),
            )
        )
        return {
            "status": "scored",
            "reactant": base["reactant"],
            "start_band_distance": start_distance,
            "products": products,
            "num_products": base.get("num_products"),
            "ranked_by": "band_distance",
        }

    async def _fallback_benchmark_transformation(
        self, smiles: str, reaction_smarts: str, max_products: int
    ) -> dict[str, Any]:
        """`moltrain.toolkit`'s composite, for when moltrain is not importable.

        Kept byte-for-byte equivalent on purpose: the harness plugin, moltrain
        and this package must all apply a reaction the same way, or a trace
        collected under one does not replay under another.
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


def build_toolkit(
    cfg: dict[str, Any], fragment_smiles: Sequence[str] | None = None
) -> Toolkit:
    return Toolkit(cfg, import_molbench(cfg), fragment_smiles)


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
