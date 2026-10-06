/**
 * MolBench: measure how a SMARTS transformation moves RDKit's heuristic
 * Synthetic Accessibility score, and show the structural change.
 *
 * The four primitives live in the `molbench` Python MCP server. This plugin
 * connects to it and adds one composite tool that runs a whole benchmark in a
 * single call: validate, score the starting material, apply the reaction,
 * score every product, and attach before/after depictions.
 * @module
 */

import { randomUUID } from 'node:crypto'
import type { Context } from '@deepseek-ai/cordis'
import { MessageId, ToolCallId } from '@deepseek-ai/dsh-llm/brand'
import type { ContextFormed, UserMessage } from '@deepseek-ai/dsh-llm/message'
import type { ContentBlock } from '@deepseek-ai/dsh-llm/types'
import * as McpClient from '@deepseek-ai/dsh-mcp-client'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolRunContext } from '@deepseek-ai/dsh-tools'
import type { JsonValue } from '@deepseek-ai/dsh-util-values'
import z from '@deepseek-ai/schemastery'

declare module '@deepseek-ai/dsh-llm/message' {
  interface MessageSourceMap {
    /** Depictions ferried out of this plugin's nested render calls. */
    molbench: { kind: 'molbench' } & ContextFormed
  }
}

/** Cordis plugin identity. */
export const name = 'molbench'

/** The composite tool registers on the shared tool registry. */
export const inject = ['tools']

/** Python server connection and benchmarking defaults. */
export interface Config {
  /** `serverName` namespace for the Python tools, as in `mcp__molbench__validate_smiles`. */
  server: string
  /** Mount the MCP client here. Set false when a separate client entry owns the connection. */
  connect: boolean
  /** Interpreter that runs the server; use the project's virtualenv Python. */
  command: string
  /** Arguments passed without a shell. */
  args: string[]
  /** Working directory for the server process. */
  cwd?: string
  /** How many products to score per transformation. */
  maxProducts: number
  /** Attach before/after depictions unless a call overrides it. */
  render: boolean
  /** Receptor .pdbqt used by dock_and_score when a call names none. */
  defaultReceptor?: string
  /** Docking engine used when a call names none. */
  defaultEngine: 'vina' | 'smina' | 'haddock3'
}

export const Config: z<Partial<Config>, Config> = z.object({
  server: z.string().pattern(/^[A-Za-z0-9_-]{1,32}$/u).default('molbench'),
  connect: z.boolean().default(true),
  command: z.string().pattern(/[^\s]/u).default('python3'),
  args: z.array(String).default(['server.py']),
  cwd: z.string(),
  maxProducts: z.number().min(1).max(10).step(1).default(3),
  render: z.boolean().default(true),
  defaultReceptor: z.string(),
  defaultEngine: z.union(['vina', 'smina', 'haddock3'] as const).default('vina'),
})

/** One scored molecule as the Python `get_molecule_metrics` tool reports it. */
interface Metrics {
  canonical_smiles: string
  sa_score: number
  exact_mw: number
  qed: number
  formula: string
}

/** The shape this tool reports per molecule. */
interface ScoredMolecule {
  smiles: string
  sa_score: number
  exact_mw: number
  qed: number
  formula: string
  svg_path?: string
}

interface ScoredProduct extends ScoredMolecule {
  sa_delta: number
  qed_delta: number
  mw_delta: number
}

const MOLECULE_PROPERTIES = {
  smiles: { type: 'string', required: true, description: 'Canonical SMILES' },
  sa_score: {
    type: 'number',
    required: true,
    description: 'Synthetic accessibility, 1 (easy) to 10 (hard)',
  },
  exact_mw: { type: 'number', required: true, description: 'Exact molecular weight' },
  qed: { type: 'number', required: true, description: 'Drug-likeness, 0 to 1' },
  formula: { type: 'string', required: true, description: 'Molecular formula' },
  svg_path: { type: 'string', description: 'Absolute path of the saved SVG, when rendered' },
} as const

const PRODUCT_PROPERTIES = {
  ...MOLECULE_PROPERTIES,
  sa_delta: {
    type: 'number',
    required: true,
    description: 'Product SA minus starting-material SA; negative is predicted easier to make',
  },
  qed_delta: { type: 'number', required: true, description: 'Product QED minus starting-material QED' },
  mw_delta: { type: 'number', required: true, description: 'Product exact MW minus starting-material MW' },
} as const

/** Unwrap the MCP canonical value into the structured object the tool returned. */
function structuredOf(value: JsonValue): Record<string, JsonValue> | undefined {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return undefined
  const structured = (value as Record<string, JsonValue>).structuredContent
  if (typeof structured === 'object' && structured !== null && !Array.isArray(structured)) {
    return structured as Record<string, JsonValue>
  }
  // A server may answer with text only; its first JSON block carries the value.
  const content = (value as Record<string, JsonValue>).content
  if (Array.isArray(content)) {
    for (const block of content) {
      if (typeof block !== 'object' || block === null || Array.isArray(block)) continue
      const text = (block as Record<string, JsonValue>).text
      if (typeof text !== 'string') continue
      try {
        const parsed: unknown = JSON.parse(text)
        if (typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed)) {
          return parsed as Record<string, JsonValue>
        }
      } catch {
        // Not this server's JSON; try the next block.
      }
    }
  }
  return undefined
}

function textOf(content: readonly ContentBlock[]): string {
  return content
    .map((block) => (block.type === 'text' ? block.text : ''))
    .filter(Boolean)
    .join(' ')
    .trim()
}

function imagesOf(content: readonly ContentBlock[]): ContentBlock[] {
  return content.filter((block) => block.type === 'image')
}

function asMetrics(structured: Record<string, JsonValue> | undefined): Metrics | undefined {
  if (!structured) return undefined
  const { canonical_smiles, sa_score, exact_mw, qed, formula } = structured
  if (
    typeof canonical_smiles !== 'string' ||
    typeof sa_score !== 'number' ||
    typeof exact_mw !== 'number' ||
    typeof qed !== 'number' ||
    typeof formula !== 'string'
  ) {
    return undefined
  }
  return { canonical_smiles, sa_score, exact_mw, qed, formula }
}

function round(value: number, places = 4): number {
  const factor = 10 ** places
  return Math.round(value * factor) / factor
}

function signed(value: number): string {
  return value >= 0 ? `+${value}` : `${value}`
}

export function apply(ctx: Context, config: Config): void {
  if (config.connect) {
    ctx.plugin(
      McpClient,
      McpClient.Config({
        transport: 'stdio',
        serverName: config.server,
        command: config.command,
        args: config.args,
        ...(config.cwd === undefined ? {} : { cwd: config.cwd }),
      }),
    )
  }

  let sequence = 0

  /** Dispatch one Python tool as a nested call under the composite's identity. */
  async function callPython(
    exec: ToolRunContext,
    rawName: string,
    args: Record<string, JsonValue>,
  ): Promise<
    | { ok: true; structured: Record<string, JsonValue> | undefined; content: readonly ContentBlock[] }
    | { ok: false; message: string }
  > {
    const result = await ctx.tools.execute({
      callId: ToolCallId(`${exec.callId}:molbench:${++sequence}`),
      rootCallId: exec.rootCallId,
      name: `mcp__${config.server}__${rawName}`,
      arguments: args,
      ...(exec.agent === undefined ? {} : { agent: exec.agent }),
      parent: exec.token,
      signal: exec.signal,
    })
    if (result.isError) {
      return { ok: false, message: textOf(result.content) || `${rawName} failed` }
    }
    return { ok: true, structured: structuredOf(result.value), content: result.content }
  }

  ctx.tools.register(
    defineTool({
      name: 'benchmark_transformation',
      description:
        'Apply a reaction SMARTS to a molecule and report how it moves the heuristic ' +
        'Synthetic Accessibility (SA) score, with QED and exact molecular weight. ' +
        'Negative sa_delta means the product is predicted easier to synthesize. ' +
        'Also attaches 2D depictions of the starting material and its products.',
      parameters: {
        smiles: { type: 'string', required: true, description: 'SMILES of the starting material' },
        reaction_smarts: {
          type: 'string',
          required: true,
          description: "Single-reactant reaction SMARTS, e.g. '[cH:1]>>[c:1]F'",
        },
        render: {
          type: 'boolean',
          description: 'Attach 2D depictions (defaults to the plugin config)',
        },
        label: { type: 'string', description: 'Filename prefix for the saved SVG files' },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            status: {
              type: 'string',
              required: true,
              enum: ['scored', 'invalid_input', 'no_products', 'server_error'],
              description: 'Outcome of the benchmark run',
            },
            detail: { type: 'string', description: 'Why a non-scored status happened' },
            reactant: {
              type: 'object',
              additionalProperties: false,
              properties: MOLECULE_PROPERTIES,
              description: 'The starting material',
            },
            products: {
              type: 'array',
              items: { type: 'object', additionalProperties: false, properties: PRODUCT_PROPERTIES },
              description: 'Scored products, most negative sa_delta first',
            },
            best_sa_delta: { type: 'number', description: 'SA change of the best product' },
            num_products: {
              type: 'number',
              description: 'Valid products the reaction produced, before scoring limits',
            },
          },
        },
        render: (args, value) => {
          const reactant = value.reactant
          if (value.status !== 'scored' || reactant === undefined) {
            const detail = value.detail === undefined ? '' : ` — ${value.detail}`
            return [{ type: 'text', text: `benchmark_transformation: ${value.status}${detail}` }]
          }
          const lines = [
            `${args.reaction_smarts} applied to ${reactant.smiles}`,
            `start: SA ${reactant.sa_score} | QED ${reactant.qed} | MW ${reactant.exact_mw} | ${reactant.formula}`,
          ]
          for (const product of value.products ?? []) {
            const direction =
              product.sa_delta < 0 ? 'easier' : product.sa_delta > 0 ? 'harder' : 'unchanged'
            lines.push(
              `product ${product.smiles}: SA ${product.sa_score} (${signed(product.sa_delta)}, ${direction})` +
                ` | QED ${product.qed} (${signed(product.qed_delta)})` +
                ` | MW ${product.exact_mw} (${signed(product.mw_delta)})`,
            )
          }
          const scoredCount = value.products?.length ?? 0
          if ((value.num_products ?? 0) > scoredCount) {
            lines.push(`(${value.num_products} valid products total; scored the first ${scoredCount})`)
          }
          return [{ type: 'text', text: lines.join('\n') }]
        },
      },
      async execute(args, exec) {
        const wantsRender = args.render ?? config.render
        const prefix = (args.label ?? 'molbench').trim() || 'molbench'

        const validation = await callPython(exec, 'validate_smiles', { smiles: args.smiles })
        if (!validation.ok) return { status: 'server_error' as const, detail: validation.message }
        const validated = validation.structured
        if (validated?.valid !== true) {
          const errors = Array.isArray(validated?.errors) ? validated.errors.join('; ') : ''
          return {
            status: 'invalid_input' as const,
            detail: errors || `RDKit rejected ${args.smiles}`,
          }
        }

        const baseline = await callPython(exec, 'get_molecule_metrics', { smiles: args.smiles })
        if (!baseline.ok) return { status: 'server_error' as const, detail: baseline.message }
        const start = asMetrics(baseline.structured)
        if (start === undefined) {
          return {
            status: 'server_error' as const,
            detail: 'get_molecule_metrics returned no usable metrics',
          }
        }

        const reaction = await callPython(exec, 'apply_smarts_reaction', {
          smiles: args.smiles,
          reaction_smarts: args.reaction_smarts,
        })
        if (!reaction.ok) return { status: 'server_error' as const, detail: reaction.message }
        const rawProducts = reaction.structured?.products
        const products = Array.isArray(rawProducts)
          ? rawProducts.filter((item): item is string => typeof item === 'string')
          : []

        const depictions: ContentBlock[] = []
        let reactantSvg: string | undefined

        if (wantsRender) {
          // Highlight the motif the reaction matches, so the change is visible.
          const reactantPattern = args.reaction_smarts.split('>>')[0] ?? ''
          const drawn = await callPython(exec, 'render_molecule_2d', {
            smiles: start.canonical_smiles,
            filename_prefix: `${prefix}_start`,
            ...(reactantPattern === '' ? {} : { highlight_smarts: reactantPattern }),
          })
          if (drawn.ok) {
            depictions.push(...imagesOf(drawn.content))
            const path = drawn.structured?.svg_path
            if (typeof path === 'string') reactantSvg = path
          }
        }

        const reactant: ScoredMolecule = {
          smiles: start.canonical_smiles,
          sa_score: start.sa_score,
          exact_mw: start.exact_mw,
          qed: start.qed,
          formula: start.formula,
          ...(reactantSvg === undefined ? {} : { svg_path: reactantSvg }),
        }

        if (products.length === 0) {
          return {
            status: 'no_products' as const,
            detail: `${args.reaction_smarts} produced no valid products from ${start.canonical_smiles}`,
            reactant,
            num_products: 0,
          }
        }

        const scored: ScoredProduct[] = []
        for (const smiles of products.slice(0, config.maxProducts)) {
          const metrics = await callPython(exec, 'get_molecule_metrics', { smiles })
          const product = asMetrics(metrics.ok ? metrics.structured : undefined)
          if (product === undefined) continue

          let productSvg: string | undefined
          if (wantsRender) {
            const drawn = await callPython(exec, 'render_molecule_2d', {
              smiles: product.canonical_smiles,
              filename_prefix: `${prefix}_product`,
            })
            if (drawn.ok) {
              depictions.push(...imagesOf(drawn.content))
              const path = drawn.structured?.svg_path
              if (typeof path === 'string') productSvg = path
            }
          }

          scored.push({
            smiles: product.canonical_smiles,
            sa_score: product.sa_score,
            exact_mw: product.exact_mw,
            qed: product.qed,
            formula: product.formula,
            sa_delta: round(product.sa_score - start.sa_score),
            qed_delta: round(product.qed - start.qed),
            mw_delta: round(product.exact_mw - start.exact_mw),
            ...(productSvg === undefined ? {} : { svg_path: productSvg }),
          })
        }

        if (scored.length === 0) {
          return {
            status: 'server_error' as const,
            detail: 'no product could be scored',
            reactant,
            num_products: products.length,
          }
        }

        scored.sort((left, right) => left.sa_delta - right.sa_delta)

        // Nested calls keep their images out of the outer result, so ferry them
        // into the agent's next request explicitly.
        if (depictions.length > 0) {
          const message: UserMessage = {
            id: MessageId(randomUUID()),
            role: 'user',
            content: [
              {
                type: 'text',
                text: `Depictions for ${args.reaction_smarts} on ${start.canonical_smiles} (starting material first).`,
              },
              ...depictions,
            ],
            source: { kind: 'molbench' },
          }
          try {
            exec.deferContext(message)
          } catch {
            // A disposed agent must not fail an otherwise complete benchmark.
          }
        }

        return {
          status: 'scored' as const,
          reactant,
          products: scored,
          best_sa_delta: scored[0]!.sa_delta,
          num_products: products.length,
        }
      },
    }),
  )

  ctx.tools.register(
    defineTool({
      name: 'dock_and_score',
      description:
        'Dock one molecule into a prepared receptor and report the 3D and 2D picture together: ' +
        'binding affinity, Ligand Efficiency, Lipophilic Ligand Efficiency, cLogP, TPSA, ' +
        'plus SA score, QED and exact molecular weight. Affinity is in kcal/mol, where more ' +
        'negative binds better, and LE below -0.3 is the usual target. Also writes an ' +
        'interactive 3D pose viewer and attaches a 2D depiction.',
      parameters: {
        smiles: { type: 'string', required: true, description: 'SMILES of the ligand to dock' },
        receptor_file: {
          type: 'string',
          description: 'Receptor .pdbqt from prepare_target_receptor; defaults to the plugin config',
        },
        engine: {
          type: 'string',
          enum: ['vina', 'smina', 'haddock3'],
          description: 'Docking engine; defaults to the plugin config',
        },
        exhaustiveness: {
          type: 'number',
          description: 'Search effort, higher is slower and more thorough (default 8)',
        },
        render: { type: 'boolean', description: 'Attach a 2D depiction (defaults to the plugin config)' },
        label: { type: 'string', description: 'Filename prefix for the saved pose files' },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            status: {
              type: 'string',
              required: true,
              enum: ['scored', 'invalid_input', 'docking_failed', 'server_error'],
              description: 'Outcome of the run',
            },
            detail: { type: 'string', description: 'Why a non-scored status happened' },
            smiles: { type: 'string', description: 'Canonical SMILES of the docked ligand' },
            engine: { type: 'string', description: 'Engine that produced the score' },
            binding_affinity: { type: 'number', description: 'Best pose affinity in kcal/mol' },
            ligand_efficiency: { type: 'number', description: 'Affinity per heavy atom; target < -0.3' },
            lipophilic_ligand_efficiency: { type: 'number', description: 'Estimated pIC50 minus cLogP' },
            estimated_pic50: { type: 'number', description: 'Score converted with 2.303RT; a ranking aid only' },
            clogp: { type: 'number', description: 'Crippen octanol-water partition coefficient' },
            tpsa: { type: 'number', description: 'Topological polar surface area' },
            sa_score: { type: 'number', description: 'Synthetic accessibility, 1 easy to 10 hard' },
            qed: { type: 'number', description: 'Drug-likeness, 0 to 1' },
            exact_mw: { type: 'number', description: 'Exact molecular weight' },
            heavy_atoms: { type: 'number', description: 'Heavy-atom count' },
            pose_strain: { type: 'number', description: 'Pose energy above its relaxed minimum, kcal/mol' },
            pose_file: { type: 'string', description: 'Docked pose as SDF' },
            complex_file: { type: 'string', description: 'Receptor plus pose as PDB' },
            viewer_file: { type: 'string', description: 'Interactive 3D viewer; open it in a browser' },
          },
        },
        render: (args, value) => {
          if (value.status !== 'scored') {
            const detail = value.detail === undefined ? '' : ` — ${value.detail}`
            return [{ type: 'text', text: `dock_and_score: ${value.status}${detail}` }]
          }
          const lines = [
            `${value.smiles} docked with ${value.engine}: ${value.binding_affinity} kcal/mol`,
            `efficiency: LE ${value.ligand_efficiency} (target < -0.3) | LLE ${value.lipophilic_ligand_efficiency}` +
              ` | estimated pIC50 ${value.estimated_pic50}`,
            `properties: cLogP ${value.clogp} | TPSA ${value.tpsa} | SA ${value.sa_score} | QED ${value.qed}` +
              ` | MW ${value.exact_mw} | ${value.heavy_atoms} heavy atoms`,
          ]
          if (value.pose_strain !== undefined) {
            lines.push(`pose strain: ${value.pose_strain} kcal/mol above its relaxed minimum`)
          }
          if (value.viewer_file !== undefined) lines.push(`3D viewer: ${value.viewer_file}`)
          return [{ type: 'text', text: lines.join('\n') }]
        },
      },
      async execute(args, exec) {
        const receptorFile = args.receptor_file ?? config.defaultReceptor
        if (receptorFile === undefined) {
          return {
            status: 'invalid_input' as const,
            detail:
              'no receptor_file given and no defaultReceptor configured; run prepare_target_receptor first',
          }
        }

        const validation = await callPython(exec, 'validate_smiles', { smiles: args.smiles })
        if (!validation.ok) return { status: 'server_error' as const, detail: validation.message }
        if (validation.structured?.valid !== true) {
          const errors = Array.isArray(validation.structured?.errors)
            ? validation.structured.errors.join('; ')
            : ''
          return { status: 'invalid_input' as const, detail: errors || `RDKit rejected ${args.smiles}` }
        }

        const docked = await callPython(exec, 'run_docking_simulation', {
          smiles: args.smiles,
          receptor_file: receptorFile,
          engine: args.engine ?? config.defaultEngine,
          ...(args.exhaustiveness === undefined ? {} : { exhaustiveness: args.exhaustiveness }),
          ...(args.label === undefined ? {} : { label: args.label }),
        })
        if (!docked.ok) return { status: 'docking_failed' as const, detail: docked.message }
        const affinity = docked.structured?.binding_affinity
        if (typeof affinity !== 'number') {
          return { status: 'docking_failed' as const, detail: 'the engine reported no binding affinity' }
        }

        const sbdd = await callPython(exec, 'get_sbdd_metrics', {
          smiles: args.smiles,
          docking_score: affinity,
        })
        if (!sbdd.ok) return { status: 'server_error' as const, detail: sbdd.message }
        const ligand = await callPython(exec, 'get_molecule_metrics', { smiles: args.smiles })
        const twoD = ligand.ok ? ligand.structured : undefined

        if (args.render ?? config.render) {
          const drawn = await callPython(exec, 'render_molecule_2d', {
            smiles: args.smiles,
            filename_prefix: `${args.label ?? 'dock'}_ligand`,
          })
          if (drawn.ok) {
            const images = imagesOf(drawn.content)
            if (images.length > 0) {
              try {
                exec.deferContext({
                  id: MessageId(randomUUID()),
                  role: 'user',
                  content: [
                    { type: 'text', text: `Docked ligand ${args.smiles} (${affinity} kcal/mol).` },
                    ...images,
                  ],
                  source: { kind: 'molbench' },
                })
              } catch {
                // A disposed agent must not fail a completed docking run.
              }
            }
          }
        }

        const num = (source: Record<string, JsonValue> | undefined, key: string): number | undefined => {
          const candidate = source?.[key]
          return typeof candidate === 'number' ? candidate : undefined
        }
        const strain = docked.structured?.pose_strain
        const poseStrain =
          typeof strain === 'object' && strain !== null && !Array.isArray(strain)
            ? num(strain as Record<string, JsonValue>, 'pose_strain')
            : undefined
        const text = (key: string): string | undefined => {
          const candidate = docked.structured?.[key]
          return typeof candidate === 'string' ? candidate : undefined
        }

        return {
          status: 'scored' as const,
          smiles:
            (typeof twoD?.canonical_smiles === 'string' ? twoD.canonical_smiles : undefined) ??
            args.smiles,
          engine: typeof docked.structured?.engine === 'string' ? docked.structured.engine : 'unknown',
          binding_affinity: affinity,
          ...(num(sbdd.structured, 'Ligand_Efficiency') === undefined
            ? {}
            : { ligand_efficiency: num(sbdd.structured, 'Ligand_Efficiency')! }),
          ...(num(sbdd.structured, 'Lipophilic_Ligand_Efficiency') === undefined
            ? {}
            : { lipophilic_ligand_efficiency: num(sbdd.structured, 'Lipophilic_Ligand_Efficiency')! }),
          ...(num(sbdd.structured, 'estimated_pIC50') === undefined
            ? {}
            : { estimated_pic50: num(sbdd.structured, 'estimated_pIC50')! }),
          ...(num(sbdd.structured, 'cLogP') === undefined ? {} : { clogp: num(sbdd.structured, 'cLogP')! }),
          ...(num(sbdd.structured, 'TPSA') === undefined ? {} : { tpsa: num(sbdd.structured, 'TPSA')! }),
          ...(num(sbdd.structured, 'heavy_atoms') === undefined
            ? {}
            : { heavy_atoms: num(sbdd.structured, 'heavy_atoms')! }),
          ...(num(twoD, 'sa_score') === undefined ? {} : { sa_score: num(twoD, 'sa_score')! }),
          ...(num(twoD, 'qed') === undefined ? {} : { qed: num(twoD, 'qed')! }),
          ...(num(twoD, 'exact_mw') === undefined ? {} : { exact_mw: num(twoD, 'exact_mw')! }),
          ...(poseStrain === undefined ? {} : { pose_strain: poseStrain }),
          ...(text('pose_file') === undefined ? {} : { pose_file: text('pose_file')! }),
          ...(text('complex_file') === undefined ? {} : { complex_file: text('complex_file')! }),
          ...(text('viewer_file') === undefined ? {} : { viewer_file: text('viewer_file')! }),
        }
      },
    }),
  )
}
