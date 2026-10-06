/**
 * Integration test for the composite tool.
 *
 * The harness itself is not installed here, so this stubs the parts of
 * `ctx.tools` the plugin uses and routes every `mcp__molbench__*` dispatch to
 * the REAL Python server over a real stdio MCP connection. The stub mirrors
 * the documented harness bridge: a canonical value of
 * `{ content, structuredContent }`, text blocks as `{ type: 'text', text }`,
 * and admitted images as attachment-backed `{ type: 'image' }` blocks.
 */

import assert from 'node:assert/strict'
import { existsSync, mkdtempSync, readdirSync, readFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { after, before, test } from 'node:test'
import { fileURLToPath } from 'node:url'

import { Client } from '@modelcontextprotocol/client'
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio'

import * as plugin from '../lib/index.js'

const PLUGIN_DIR = dirname(dirname(fileURLToPath(import.meta.url)))
const PROJECT_DIR = dirname(PLUGIN_DIR)
const PYTHON = join(PROJECT_DIR, '.venv', 'bin', 'python')
const BENZENE = 'c1ccccc1'
const FLUORINATE = '[cH:1]>>[c:1]F'

let client
let vizDir
let tool
let dockTool

/** Map MCP content blocks the way the harness bridge does. */
function toBlocks(content = []) {
  return content.map((block) =>
    block.type === 'image'
      ? { type: 'image', attachment: { id: 'stub-attachment', mimeType: block.mimeType ?? block.mime_type } }
      : { type: 'text', text: block.text ?? '' },
  )
}

function makeContext() {
  const registered = new Map()
  const ctx = {
    plugin() {
      throw new Error('the test configures connect:false, so no child plugin should mount')
    },
    tools: {
      register(definition) {
        registered.set(definition.name, definition)
        return () => registered.delete(definition.name)
      },
      async execute({ name, arguments: args }) {
        const prefix = 'mcp__molbench__'
        assert.ok(name.startsWith(prefix), `expected a molbench MCP call, got ${name}`)
        try {
          const result = await client.callTool({ name: name.slice(prefix.length), arguments: args })
          const content = toBlocks(result.content)
          if (result.isError) {
            return { isError: true, error: { message: content.map((b) => b.text).join(' ') }, content }
          }
          return {
            isError: false,
            value: {
              content: result.content,
              ...(result.structuredContent === undefined
                ? {}
                : { structuredContent: result.structuredContent }),
            },
            content,
          }
        } catch (error) {
          return {
            isError: true,
            error: { message: String(error) },
            content: [{ type: 'text', text: String(error) }],
          }
        }
      },
    },
  }
  return { ctx, registered }
}

/** One execution context, collecting whatever the tool defers. */
function makeExec() {
  const deferred = []
  return {
    exec: {
      callId: 'test-call-1',
      rootCallId: 'test-call-1',
      name: 'benchmark_transformation',
      arguments: {},
      token: Symbol('token'),
      signal: new AbortController().signal,
      deferContext: (message) => deferred.push(message),
      concludeTurn: () => {},
    },
    deferred,
  }
}

before(async () => {
  vizDir = mkdtempSync(join(tmpdir(), 'molbench-viz-'))
  const transport = new StdioClientTransport({
    command: PYTHON,
    args: [join(PROJECT_DIR, 'server.py')],
    cwd: PROJECT_DIR,
    env: { ...process.env, MOLBENCH_VIZ_DIR: vizDir },
  })
  client = new Client({ name: 'molbench-integration-test', version: '0.0.0' })
  await client.connect(transport)

  const { ctx, registered } = makeContext()
  plugin.apply(ctx, plugin.Config({ connect: false, maxProducts: 3 }))
  tool = registered.get('benchmark_transformation')
  dockTool = registered.get('dock_and_score')
})

after(async () => {
  await client?.close()
})

test('registers both composite tools', () => {
  assert.ok(tool, 'benchmark_transformation should be registered')
  assert.match(tool.description, /Synthetic Accessibility/)
  assert.ok(dockTool, 'dock_and_score should be registered')
  assert.match(dockTool.description, /Ligand Efficiency/)
})

test('scores a fluorination and attaches depictions', async () => {
  const args = { smiles: BENZENE, reaction_smarts: FLUORINATE, label: 'benzene_f' }
  const { exec, deferred } = makeExec()
  const value = await tool.execute(args, exec)

  assert.equal(value.status, 'scored')
  assert.equal(value.reactant.smiles, BENZENE)
  assert.equal(value.num_products, 1)
  assert.equal(value.products.length, 1)

  const [product] = value.products
  assert.equal(product.smiles, 'Fc1ccccc1')
  // Deltas must be consistent with the two absolute scores.
  assert.ok(Math.abs(product.sa_delta - (product.sa_score - value.reactant.sa_score)) < 1e-6)
  assert.ok(Math.abs(product.mw_delta - (product.exact_mw - value.reactant.exact_mw)) < 1e-6)
  // Fluorine for hydrogen adds about 18 Da.
  assert.ok(product.mw_delta > 17.9 && product.mw_delta < 18.1, `mw_delta was ${product.mw_delta}`)
  assert.equal(value.best_sa_delta, product.sa_delta)

  // Both molecules were drawn: SVGs on disk, PNGs ferried into context.
  assert.ok(value.reactant.svg_path.startsWith(vizDir))
  assert.ok(product.svg_path.startsWith(vizDir))
  assert.equal(readdirSync(vizDir).filter((f) => f.endsWith('.svg')).length, 2)

  assert.equal(deferred.length, 1)
  const images = deferred[0].content.filter((block) => block.type === 'image')
  assert.equal(images.length, 2)
  assert.equal(images[0].attachment.mimeType, 'image/png')
  assert.equal(deferred[0].source.kind, 'molbench')
  assert.equal(deferred[0].role, 'user')

  const rendered = tool.output.render(args, value).map((block) => block.text).join('\n')
  assert.match(rendered, /start: SA /)
  assert.match(rendered, /product Fc1ccccc1: SA .*(easier|harder|unchanged)/)
})

test('reports invalid input instead of throwing', async () => {
  const { exec } = makeExec()
  const value = await tool.execute(
    { smiles: 'c1ccccc', reaction_smarts: FLUORINATE, render: false },
    exec,
  )
  assert.equal(value.status, 'invalid_input')
  assert.match(value.detail, /unclosed ring/)
})

test('reports a transformation that matches nothing', async () => {
  const { exec } = makeExec()
  const value = await tool.execute(
    { smiles: BENZENE, reaction_smarts: '[N:1]>>[N:1]C', render: false },
    exec,
  )
  assert.equal(value.status, 'no_products')
  assert.equal(value.num_products, 0)
})

test('surfaces a server-side reaction error', async () => {
  const { exec } = makeExec()
  const value = await tool.execute(
    { smiles: BENZENE, reaction_smarts: 'not a reaction', render: false },
    exec,
  )
  assert.equal(value.status, 'server_error')
  assert.match(value.detail, /reaction_smarts/)
})

test('render:false skips drawing entirely', async () => {
  const before = readdirSync(vizDir).length
  const { exec, deferred } = makeExec()
  const value = await tool.execute(
    { smiles: BENZENE, reaction_smarts: FLUORINATE, render: false },
    exec,
  )
  assert.equal(value.status, 'scored')
  assert.equal(value.reactant.svg_path, undefined)
  assert.equal(value.products[0].svg_path, undefined)
  assert.equal(deferred.length, 0)
  assert.equal(readdirSync(vizDir).length, before)
})

// --- dock_and_score. Needs the docking engines and a prepared 3PTB receptor.

const RECEPTOR = join(PROJECT_DIR, 'receptors', '3PTB_BEN.pdbqt')
const hasReceptor = existsSync(RECEPTOR) && existsSync(join(PROJECT_DIR, '.conda-dock', 'bin', 'vina'))

test('dock_and_score reports 3D and 2D metrics together', { skip: !hasReceptor }, async () => {
  const args = { smiles: 'c1ccc(cc1)C(=N)N', receptor_file: RECEPTOR, label: 'node_bzd', render: true }
  const { exec, deferred } = makeExec()
  const value = await dockTool.execute(args, exec)

  assert.equal(value.status, 'scored', value.detail)
  assert.equal(value.engine, 'vina')
  // Benzamidine in trypsin scores near -6 kcal/mol.
  assert.ok(value.binding_affinity < -4.5 && value.binding_affinity > -8,
    `affinity was ${value.binding_affinity}`)
  // Ligand efficiency must equal affinity per heavy atom.
  assert.ok(Math.abs(value.ligand_efficiency - value.binding_affinity / value.heavy_atoms) < 1e-3)
  // The 2D side came through as well.
  assert.ok(value.sa_score > 0 && value.qed > 0 && value.exact_mw > 100)
  assert.ok(typeof value.clogp === 'number' && value.tpsa > 0)
  // Pose files exist, including the interactive viewer.
  for (const key of ['pose_file', 'complex_file', 'viewer_file']) {
    assert.ok(existsSync(value[key]), `${key} should exist`)
  }
  assert.match(readFileSync(value.viewer_file, 'utf8'), /3Dmol/)

  // The 2D depiction was ferried into context.
  assert.equal(deferred.length, 1)
  assert.equal(deferred[0].content.filter((b) => b.type === 'image').length, 1)

  const rendered = dockTool.output.render(args, value).map((b) => b.text).join('\n')
  assert.match(rendered, /LE -?\d/)
  assert.match(rendered, /3D viewer:/)
})

test('dock_and_score refuses an invalid ligand', { skip: !hasReceptor }, async () => {
  const { exec } = makeExec()
  const value = await dockTool.execute(
    { smiles: 'c1ccccc', receptor_file: RECEPTOR, render: false }, exec)
  assert.equal(value.status, 'invalid_input')
  assert.match(value.detail, /unclosed ring/)
})

test('dock_and_score explains a missing receptor', async () => {
  const { exec } = makeExec()
  const value = await dockTool.execute({ smiles: 'CCO', render: false }, exec)
  assert.equal(value.status, 'invalid_input')
  assert.match(value.detail, /defaultReceptor/)
})
