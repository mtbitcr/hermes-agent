import { execFileSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { runInNewContext } from 'node:vm'

import { afterAll, describe, expect, it } from 'vitest'

const kitDir = new URL('../scripts/design_compare/', import.meta.url)
const capturePath = new URL('capture.mjs', kitDir).href
const probeSource = readFileSync(new URL('probe.js', kitDir), 'utf8')
const kit = await import(/* @vite-ignore */ capturePath)
const probe = runInNewContext(probeSource, {})

const scratch = mkdtempSync(join(tmpdir(), 'design-compare-test-'))
afterAll(() => rmSync(scratch, { force: true, recursive: true }))

type FakeNode = Record<string, any>
const text = (nodeValue: string) => ({ nodeType: 3, nodeValue })

function el(localName: string, attrs: Record<string, string>, ...kids: (FakeNode | string)[]): FakeNode {
  const node: FakeNode = { localName, nodeType: 1, parentElement: null, children: [] as FakeNode[] }
  node.getAttribute = (n: string) => attrs[n] ?? null
  node.childNodes = kids.map(k => (typeof k === 'string' ? text(k) : k))
  node.children = node.childNodes.filter((k: FakeNode) => k.nodeType === 1)
  node.children.forEach((k: FakeNode) => { k.parentElement = node })

  node.querySelector = (): FakeNode | null => {
    const walk = (n: FakeNode): FakeNode | null => {
      for (const k of n.children) {
        if (/^h[1-6]$/.test(k.localName)) {return k}
        const hit = walk(k)

        if (hit) {return hit}
      }

      return null
    }

    return walk(node)
  }

  node.textContent = kids.map(k => (typeof k === 'string' ? k : k.textContent)).join('')

  return node
}

describe('key building', () => {
  it('normalizes with NFKC, collapsed whitespace, lower case and a character limit', () => {
    expect(probe.normalize('  Ｆｕｌｌ\n  WIDTH\tText ', 60)).toBe('full width text')
    expect(probe.normalize('x'.repeat(100), 60)).toHaveLength(60)
  })

  it('builds chain/role:"name"#n keys in document order', () => {
    const h1 = el('h1', {}, 'Hello \n World')
    const buyA = el('a', {}, 'Buy')
    const buyB = el('a', {}, ' buy ')
    const close = el('button', { 'aria-label': 'Close' }, 'x')
    const logo = el('img', { alt: 'Logo' })
    const longHeading = el('h2', {}, 'A'.repeat(60))
    const section = el('section', {}, longHeading, buyA, buyB, close)
    const header = el('header', {}, h1, logo)
    const main = el('main', {}, section)
    el('body', {}, header, main)

    const keys = probe.assignKeys([h1, logo, longHeading, buyA, buyB, close, section])
    expect(keys).toEqual([
      'header[hello world]/h1:"hello world"#1',
      'header[hello world]/img:"logo"#1',
      `main[${'a'.repeat(40)}] > section[${'a'.repeat(40)}]/h2:"${'a'.repeat(60)}"#1`,
      `main[${'a'.repeat(40)}] > section[${'a'.repeat(40)}]/a:"buy"#1`,
      `main[${'a'.repeat(40)}] > section[${'a'.repeat(40)}]/a:"buy"#2`,
      `main[${'a'.repeat(40)}] > section[${'a'.repeat(40)}]/button:"close"#1`,
      `main[${'a'.repeat(40)}]/section:""#1`,
    ])
  })

  it('prefers an explicit role to the tag name', () => {
    const d = el('div', { role: 'alert' }, 'Saved')
    el('body', {}, d)
    expect(probe.assignKeys([d])).toEqual(['/alert:"saved"#1'])
  })
})

const SHA = 'a'.repeat(64)

// Literal fixture of design_compare_bundle_v1 (C1 owns the schema; copied here field for field).
const BUNDLE_V1 = {
  schema: 'design_compare_bundle_v1',
  fields: {
    schema: 'const', kit_sha256: 'sha256', chromium_version: 'string', run_id: 'string', candidate_digest: 'string',
    page: 'string', width: [1280, 375], page_height: 'int', image: 'string', blocked_requests: ['string'],
    masks: [{ x: 'number', y: 'number', width: 'number', height: 'number' }],
    elements: [{
      key: 'string', tag: 'string', text: 'string', pinned: 'boolean',
      box: { x: 'number', y: 'number', width: 'number', height: 'number' },
      type: { family: 'string', size: 'number', weight: 'number', line_height: 'number|normal', face_loaded: 'boolean' },
      colour: {
        color: 'rgba', background_color: 'rgba', border_top_color: 'rgba', border_right_color: 'rgba',
        border_bottom_color: 'rgba', border_left_color: 'rgba', text_decoration_color: 'rgba', fill: 'rgba', stroke: 'rgba',
      },
    }],
  },
}

function conforms(value: any, spec: any, path = '$'): string[] {
  const bad = (why: string) => [`${path}: ${why}`]

  if (Array.isArray(spec)) {
    if (typeof spec[0] === 'number') {return spec.includes(value) ? [] : bad(`not one of ${spec}`)}

    if (!Array.isArray(value)) {return bad('not an array')}

    return value.flatMap((v, i) => conforms(v, spec[0], `${path}[${i}]`))
  }

  if (typeof spec === 'object') {
    if (value === null || typeof value !== 'object') {return bad('not an object')}
    const extra = Object.keys(value).filter(k => !(k in spec)).map(k => `${path}.${k}: unexpected`)

    return [...extra, ...Object.keys(spec).flatMap(k => (k in value ? conforms(value[k], spec[k], `${path}.${k}`) : bad(`missing ${k}`)))]
  }

  const ok: Record<string, (v: any) => boolean> = {
    string: v => typeof v === 'string',
    number: v => typeof v === 'number' && Number.isFinite(v),
    int: v => Number.isInteger(v),
    boolean: v => typeof v === 'boolean',
    sha256: v => typeof v === 'string' && /^[0-9a-f]{64}$/.test(v),
    'number|normal': v => v === 'normal' || (typeof v === 'number' && Number.isFinite(v)),
    rgba: v => v === null || (Array.isArray(v) && v.length === 4 && v.every(n => typeof n === 'number')),
    const: v => v === BUNDLE_V1.schema,
  }

  return ok[spec](value) ? [] : bad(`not ${spec}`)
}

const rgba = [1, 2, 3, 1]

const sampleElement = {
  key: 'header/h1:"hi"#1', tag: 'h1', text: 'hi', pinned: false,
  box: { x: 0, y: 8, width: 100, height: 30 },
  type: { family: 'geist', size: 16, weight: 500, line_height: 'normal', face_loaded: true },
  colour: {
    color: rgba, background_color: [0, 0, 0, 0], border_top_color: null, border_right_color: null,
    border_bottom_color: null, border_left_color: null, text_decoration_color: rgba, fill: null, stroke: null,
  },
}

const bundleInput = {
  kitSha256: SHA, chromiumVersion: '1.2.3.4', runId: 'run-7', candidateDigest: 'digest', page: '/', width: 1280,
  pageHeight: 800, image: 'index-1280.png', blockedRequests: ['https://beacon.example/x'],
  measurement: { elements: [sampleElement], masks: [{ x: 0, y: 0, width: 5, height: 5 }] },
}

describe('bundle', () => {
  it('matches design_compare_bundle_v1 field for field', () => {
    const bundle = JSON.parse(JSON.stringify(kit.buildBundle(bundleInput)))
    expect(conforms(bundle, BUNDLE_V1.fields)).toEqual([])
    expect(bundle.schema).toBe('design_compare_bundle_v1')
    expect(bundle.elements[0].key).toBe(sampleElement.key)
  })

  it('the fixture check rejects a missing, extra or mistyped field', () => {
    const good = JSON.parse(JSON.stringify(kit.buildBundle(bundleInput)))
    const { run_id: _dropped, ...missing } = good
    expect(conforms(missing, BUNDLE_V1.fields)).not.toEqual([])
    expect(conforms({ ...good, surprise: 1 }, BUNDLE_V1.fields)).not.toEqual([])
    expect(conforms({ ...good, width: 1024 }, BUNDLE_V1.fields)).not.toEqual([])
  })
})

describe('requests and Chromium pin', () => {
  const origin = 'http://127.0.0.1:5000'

  it('classifies anything outside the local server as blocked', () => {
    for (const url of [`${origin}/`, `${origin}/a.css?x=1`, 'data:text/css,a{}', 'blob:http://127.0.0.1:5000/1', 'about:blank']) {
      expect(kit.classifyRequest(url, origin), url).toBe('allowed')
    }

    for (const url of [
      'https://analytics.example/beacon.js', 'http://127.0.0.1:5001/', 'http://127.0.0.1:5000.evil.test/',
      'file:///etc/passwd', 'ws://127.0.0.1:5000/', 'not a url',
    ]) {
      expect(kit.classifyRequest(url, origin), url).toBe('blocked')
    }
  })

  it('fails closed when a prototype asked for anything outside itself', () => {
    expect(() => kit.assertSelfContained([])).not.toThrow()
    expect(() => kit.assertSelfContained(['https://fonts.example/f.woff2'])).toThrow('not self-contained')
  })

  const binary = join(scratch, 'fake-chrome')
  writeFileSync(binary, 'fake binary')
  const sha = createHash('sha256').update('fake binary').digest('hex')
  const deps = { version: () => 'Chromium 153.0.8010.0' }

  it('starts only on the pinned version and binary hash', () => {
    expect(kit.verifyChromium(binary, { version: '153.0.8010.0', sha256: sha }, deps)).toBe('153.0.8010.0')
  })

  it('refuses a missing, empty or mismatched hash and another version', () => {
    expect(() => kit.verifyChromium(binary, { version: '153.0.8010.0' }, deps)).toThrow('hash')
    expect(() => kit.verifyChromium(binary, { version: '153.0.8010.0', sha256: '' }, deps)).toThrow('hash')
    expect(() => kit.verifyChromium(binary, { version: '153.0.8010.0', sha256: 'b'.repeat(64) }, deps)).toThrow('does not match')
    expect(() => kit.verifyChromium(binary, { version: '154.0.0.0', sha256: sha }, deps)).toThrow('version')
    expect(() => kit.verifyChromium(join(scratch, 'absent'), { version: '153.0.8010.0', sha256: sha }, deps)).toThrow()
  })

  it('ships a pinned version and refuses to start while its hash is unset', () => {
    expect(kit.CHROMIUM_PIN.version).toBe('153.0.8010.0')
    expect(() => kit.verifyChromium(binary, kit.CHROMIUM_PIN, deps)).toThrow()
  })
})

describe.skipIf(process.env.DESIGN_COMPARE_REAL_CHROMIUM !== '1')('real Chromium (opt-in)', () => {
  const page = (anim: string, smoothing: string) => `<!doctype html><meta charset=utf-8>
<style>body{margin:0;font:20px sans-serif;-webkit-font-smoothing:${smoothing}}
@keyframes spin{to{transform:rotate(360deg)}}
#box{width:80px;height:80px;background:#369;animation:spin ${anim} linear infinite}
h1{margin:16px}</style><h1>Tiny page</h1><div id=box></div><p>Some text</p>`

  it('gives identical bundles and images despite animation and font smoothing', async () => {
    const chromium = process.env.CHROME_PATH as string
    const version = /\d+(\.\d+){3}/.exec(execFileSync(chromium, ['--version'], { encoding: 'utf8' }))![0]
    const pin = { version, sha256: createHash('sha256').update(readFileSync(chromium)).digest('hex') }
    const out: string[] = []

    for (const [name, anim, smoothing] of [['a', '1s', 'antialiased'], ['b', '3.7s', 'subpixel-antialiased']]) {
      const site = join(scratch, `site-${name}`)
      mkdirSync(site)
      writeFileSync(join(site, 'index.html'), page(anim, smoothing))
      const outDir = join(scratch, `out-${name}`)
      await kit.capture({
        side: 'candidate', siteDir: site, pages: ['/'], outDir, chromium, pin, runId: 'run-1', candidateDigest: 'd1',
      })
      out.push(outDir)
    }

    for (const file of ['index-1280.json', 'index-375.json', 'index-1280.png', 'index-375.png']) {
      expect(readFileSync(join(out[0], file)).equals(readFileSync(join(out[1], file))), file).toBe(true)
    }

    const bundle = JSON.parse(readFileSync(join(out[0], 'index-1280.json'), 'utf8'))
    expect(conforms(bundle, BUNDLE_V1.fields)).toEqual([])
    expect(bundle.elements.length).toBeGreaterThan(2)
  }, 120_000)
})
