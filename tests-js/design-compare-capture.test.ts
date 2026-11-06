import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { createSocket } from 'node:dgram'
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { createServer as createTcpServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { runInNewContext } from 'node:vm'
import { inflateSync } from 'node:zlib'

import { afterAll, describe, expect, it } from 'vitest'

const kitDir = new URL('../scripts/design_compare/', import.meta.url)
const capturePath = new URL('capture.mjs', kitDir).href
const captureFile = fileURLToPath(new URL('capture.mjs', kitDir))
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

// Port of the key rule in tools/design_compare_judge.py _entries (C1), applied to emitted elements.
function judgeKeys(elements: any[]) {
  const norm = (t: string) => (t ?? '').normalize('NFKC').split(/\s+/).filter(Boolean).join(' ').toLowerCase()
  const name = (t: string) => Array.from(norm(t)).slice(0, 60).join('')
  const seen = new Map<string, number>()

  return elements.map(e => {
    const chain = e.landmarks.map(([tag, heading]: [string, string]) => (norm(heading) ? `${tag}:${Array.from(norm(heading)).slice(0, 40).join('')}` : tag)).join('/')
    const role = e.role || e.tag
    const n = name(e.label) || name(e.alt) || name(e.text)
    const id = `${chain}\u0000${role}\u0000${n}`
    seen.set(id, (seen.get(id) ?? 0) + 1)

    return `${chain}/${role}:"${n}"#${seen.get(id)}`
  })
}

describe('key building', () => {
  it('normalizes with NFKC, collapsed whitespace, lower case and a character limit', () => {
    expect(probe.normalize('  Ｆｕｌｌ\n  WIDTH\tText ', 60)).toBe('full width text')
    expect(probe.normalize('x'.repeat(100), 60)).toHaveLength(60)
  })

  it('records the ancestor landmarks with their first heading, so the judge builds chain/role:"name"#n keys in document order', () => {
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

    const inner = [['main', 'a'.repeat(40)], ['section', 'a'.repeat(40)]]
    expect(probe.landmarksOf(h1)).toEqual([['header', 'hello world']])
    expect(probe.landmarksOf(longHeading)).toEqual(inner)
    expect(probe.landmarksOf(section)).toEqual([['main', 'a'.repeat(40)]])
    expect(probe.landmarksOf(header)).toEqual([])

    const asElement = (node: FakeNode, extra: Record<string, string>) => ({ landmarks: probe.landmarksOf(node), tag: node.localName, role: node.localName, label: '', alt: '', text: '', ...extra })

    expect(judgeKeys([
      asElement(h1, { text: 'hello world' }), asElement(logo, { alt: 'Logo' }), asElement(longHeading, { text: 'a'.repeat(60) }),
      asElement(buyA, { text: 'buy' }), asElement(buyB, { text: 'buy' }), asElement(close, { label: 'Close', text: 'x' }), asElement(section, {}),
    ])).toEqual([
      'header:hello world/h1:"hello world"#1',
      'header:hello world/img:"logo"#1',
      `main:${'a'.repeat(40)}/section:${'a'.repeat(40)}/h2:"${'a'.repeat(60)}"#1`,
      `main:${'a'.repeat(40)}/section:${'a'.repeat(40)}/a:"buy"#1`,
      `main:${'a'.repeat(40)}/section:${'a'.repeat(40)}/a:"buy"#2`,
      `main:${'a'.repeat(40)}/section:${'a'.repeat(40)}/button:"close"#1`,
      `main:${'a'.repeat(40)}/section:""#1`,
    ])
  })

  it('continues the landmark chain through shadow hosts and frame elements', () => {
    const inShadow = el('h2', {}, 'Inner')
    const host = el('div', {})
    el('body', {}, el('main', {}, host))
    inShadow.getRootNode = () => ({ host })
    const frame = el('iframe', {})
    el('body', {}, el('section', {}, frame))
    const inFrame = el('h3', {}, 'Framed')
    inFrame.ownerDocument = { defaultView: { frameElement: frame } }
    expect(probe.landmarksOf(inShadow)).toEqual([['main', '']])
    expect(probe.landmarksOf(inFrame)).toEqual([['section', '']])
  })
})

describe('first font family', () => {
  // Mirrors _first_family in tools/design_compare_judge.py: quote and escape aware, unquoted, lower case.
  it('keeps a comma inside a quoted family and splits only on the comma that ends the first family', () => {
    expect(probe.firstFamily('"Acme,Display", serif')).toBe('acme,display')
    expect(probe.firstFamily('"Acme,Other", serif')).toBe('acme,other')
    expect(probe.firstFamily("'Acme, Display' , serif")).toBe('acme, display')
    expect(probe.firstFamily('Georgia, "Times New Roman", serif')).toBe('georgia')
    expect(probe.firstFamily('  Geist Variable ,sans-serif')).toBe('geist variable')
    expect(probe.firstFamily('"Acme"')).toBe('acme')
    expect(probe.firstFamily('')).toBe('')
  })

  it('honours backslash escapes and the other quote character inside a quoted family', () => {
    expect(probe.firstFamily('"Ac\\"me,X", serif')).toBe('ac\\"me,x')
    expect(probe.firstFamily('"It\'s,fine", serif')).toBe("it's,fine")
    expect(probe.firstFamily('Acme\\,X, serif')).toBe('acme\\,x')
  })

  it('names a FontFace family as one name, so its commas are never list separators', () => {
    expect(probe.familyName('Acme,Display')).toBe('acme,display')
    expect(probe.familyName('"Acme,Display"')).toBe('acme,display')
    expect(probe.familyName(" 'Acme,Other' ")).toBe('acme,other')
    expect(probe.familyName('Acme')).toBe('acme')
    expect(probe.familyName('Acme,Display')).toBe(probe.firstFamily('"Acme,Display", serif'))
    expect(probe.familyName('Acme,Display')).not.toBe(probe.firstFamily('"Acme,Other", serif'))
  })

  it('gives different identities to different comma-containing names and the same one to equal names', () => {
    const family = (v: string) => probe.firstFamily(v)
    expect(family('"Acme,Display", serif')).not.toBe(family('"Acme,Other", serif'))
    expect(family('"Acme,Display", serif')).toBe(family("'ACME,display', monospace"))
    expect(family('"Acme,Display"')).toBe(probe.firstFamily('"Acme,Display", serif'))
  })
})

describe('probe visibility and motion', () => {
  const node = (opacity: string, parent: FakeNode | null = null): FakeNode => ({ opacity, parentElement: parent })
  const style = (n: FakeNode) => ({ opacity: n.opacity })

  it('treats an element under an opacity-zero ancestor as invisible, and keeps visible controls', () => {
    const hidden = node('0')
    const child = node('1', hidden)
    const grandchild = node('0.5', child)
    const control = node('1', node('0.5'))
    expect(probe.hiddenByOpacity(hidden, style)).toBe(true)
    expect(probe.hiddenByOpacity(child, style)).toBe(true)
    expect(probe.hiddenByOpacity(grandchild, style)).toBe(true)
    expect(probe.hiddenByOpacity(control, style)).toBe(false)
  })

  const animation = (endTime: number, failFinish = false, stays = false) => {
    const a: Record<string, any> = { calls: [] as string[], playState: 'running', effect: { getComputedTiming: () => ({ endTime }) } }

    a.finish = () => { a.calls.push('finish');

 if (failFinish) {throw new Error('InvalidStateError')} a.playState = 'finished' }

    a.cancel = () => { a.calls.push('cancel');

 if (!stays) {a.playState = 'idle'} }

    return a
  }

  const settleWith = (...list: Record<string, any>[]) => {
    const live = () => list.filter(a => a.playState !== 'idle')
    const { settle } = runInNewContext(probeSource, { window: {}, scrollX: 0, scrollY: 0, document: { createElement: () => ({}), getAnimations: live, documentElement: {}, body: { querySelectorAll: () => [] } } })
    settle()

    return list.map(a => a.calls)
  }

  it('finishes finite animations, cancels infinite ones and fails when one is still running', () => {
    expect(settleWith(animation(500), animation(Infinity), animation(500, true))).toEqual([['finish'], ['cancel'], ['finish', 'cancel']])
    expect(() => settleWith(animation(Infinity, false, true))).toThrow('still running')
  })
})

const SHA = 'a'.repeat(64)

// C1 contract, copied literally from tools/design_compare_judge.py at ca87885d13bb4c83638cc8b1f655b70c6c55a1d6
// (blob 593a770a561b7eac7448ffb63586d3a7b0386809). WIDTHS, COLOUR_PROPERTIES and BOX_FIELDS are the constants those four depend on.
const SCHEMA = 'design_compare_bundle_v1'
const WIDTHS = [1280, 375]

const COLOUR_PROPERTIES = [
  'color', 'background-color', 'border-top-color', 'border-right-color', 'border-bottom-color',
  'border-left-color', 'text-decoration-color', 'fill', 'stroke',
]

const BUNDLE_FIELDS = ['schema', 'page', 'width', 'elements']
const ELEMENT_FIELDS = ['landmarks', 'tag', 'role', 'label', 'alt', 'text', 'parent', 'fixed', 'box', 'style']
const BOX_FIELDS = ['x', 'y', 'width', 'height']
const STYLE_FIELDS = ['font-family', 'font-face', 'font-size', 'font-weight', 'line-height', ...COLOUR_PROPERTIES]

// Port of _validate and _require of the same file: the first problem wins, only presence is checked, and any extra field
// (bundle metadata, for example) is tolerated.
function judgeValidate(bundle: any): string | null {
  const isObject = (v: any) => v !== null && typeof v === 'object' && !Array.isArray(v)

  const require = (obj: any, fields: string[], where: string) => {
    if (!isObject(obj)) {return `${where} is not an object`}

    for (const f of fields) {
      if (!Object.hasOwn(obj, f)) {return `${where} is missing field ${f}`}
    }

    return null
  }

  if (!isObject(bundle) || bundle.schema !== SCHEMA) {return `unknown bundle schema, expected ${SCHEMA}`}
  const top = require(bundle, BUNDLE_FIELDS, 'bundle')

  if (top) {return top}

  if (!WIDTHS.includes(bundle.width)) {return `bundle width ${bundle.width} is not one of ${WIDTHS}`}

  for (const [i, element] of Array.from(bundle.elements as any[]).entries()) {
    const problem = require(element, ELEMENT_FIELDS, `element ${i}`) ?? require(element.box, BOX_FIELDS, `element ${i} box`)
      ?? require(element.style, STYLE_FIELDS, `element ${i} style`)

    if (problem) {return problem}
  }

  return null
}

// The value shapes compare_bundles, _entries, _shift and _style_differences of the same file rely on after _validate has passed.
function judgeShapeProblems(bundle: any): string[] {
  const out: string[] = []
  const num = (v: any) => typeof v === 'number' && Number.isFinite(v)
  const str = (v: any) => typeof v === 'string'

  const check = (ok: boolean, why: string) => { if (!ok) {out.push(why)} }

  check(str(bundle.page), 'page is a string')
  check(Array.isArray(bundle.elements), 'elements is a list')

  for (const [i, e] of (bundle.elements as any[]).entries()) {
    const at = `element ${i}`
    check(Array.isArray(e.landmarks) && e.landmarks.every((l: any) => Array.isArray(l) && l.length === 2 && str(l[0]) && str(l[1])), `${at} landmarks are [tag, heading] pairs`)

    for (const f of ['tag', 'role', 'label', 'alt', 'text']) {check(str(e[f]), `${at} ${f} is a string`)}

    check(e.tag !== '' && e.role !== '', `${at} has a tag and a role`)
    check(e.parent === null || Number.isInteger(e.parent), `${at} parent is an identity or null`)
    check(typeof e.fixed === 'boolean', `${at} fixed is a boolean`)

    for (const f of BOX_FIELDS) {check(num(e.box[f]), `${at} box.${f} is a number`)}

    const s = e.style
    check(str(s['font-family']) && s['font-family'] !== '', `${at} font-family is a string`)
    check(str(s['font-face']), `${at} font-face is a string`)
    check(num(s['font-size']), `${at} font-size is a number`)
    check(num(s['font-weight']), `${at} font-weight is a number`)
    check(s['line-height'] === 'normal' || num(s['line-height']), `${at} line-height is a number or normal`)

    for (const f of COLOUR_PROPERTIES) {check(s[f] === null || (Array.isArray(s[f]) && s[f].length === 4 && s[f].every(num)), `${at} ${f} is null or [r, g, b, a]`)}
  }

  return out
}

// The emitted contract: the judge's fields and nothing else on an element, its box and its style; metadata only on the bundle.
function contractProblems(bundle: any): string[] {
  const exact = (obj: any, fields: string[], where: string) =>
    Object.keys(obj).filter(k => !fields.includes(k)).map(k => `${where}.${k}: unexpected`)

  const first = judgeValidate(bundle)

  if (first) {return [first]}

  return [
    ...judgeShapeProblems(bundle),
    ...bundle.elements.flatMap((e: any, i: number) => [
      ...exact(e, ELEMENT_FIELDS, `element ${i}`), ...exact(e.box, BOX_FIELDS, `element ${i} box`), ...exact(e.style, STYLE_FIELDS, `element ${i} style`),
    ]),
    ...(['kit_sha256', 'chromium_version', 'run_id', 'candidate_digest', 'page_height', 'image', 'blocked_requests', 'masks'] as const)
      .filter(k => !(k in bundle)).map(k => `bundle is missing metadata ${k}`),
  ]
}

const rgba = [1, 2, 3, 1]

const sampleElement = {
  landmarks: [['header', 'hi']], tag: 'h1', role: 'h1', label: '', alt: '', text: 'hi', parent: 3, fixed: false,
  box: { x: 0, y: 8, width: 100, height: 30 },
  style: {
    'font-family': '"Acme,Display", serif', 'font-face': 'acme,display', 'font-size': 16, 'font-weight': 500, 'line-height': 'normal',
    color: rgba, 'background-color': [0, 0, 0, 0], 'border-top-color': null, 'border-right-color': null, 'border-bottom-color': null,
    'border-left-color': null, 'text-decoration-color': rgba, fill: null, stroke: null,
  },
}

const bundleInput = {
  kitSha256: SHA, chromiumVersion: '1.2.3.4', runId: 'run-7', candidateDigest: 'digest', page: '/', width: 1280,
  pageHeight: 800, image: 'index-1280.png', blockedRequests: ['https://beacon.example/x'],
  measurement: { elements: [sampleElement], masks: [{ x: 0, y: 0, width: 5, height: 5 }] },
}

describe('bundle', () => {
  it('pins the copied C1 contract constants', () => {
    expect(SCHEMA).toBe('design_compare_bundle_v1')
    expect(STYLE_FIELDS).toHaveLength(14)
    expect(ELEMENT_FIELDS).toHaveLength(10)
  })

  it('a constructed bundle satisfies the C1 validator, the value shapes the comparator relies on, and keeps the capture metadata', () => {
    const bundle = JSON.parse(JSON.stringify(kit.buildBundle(bundleInput)))
    expect(contractProblems(bundle)).toEqual([])
    expect(bundle).toMatchObject({ schema: SCHEMA, kit_sha256: SHA, chromium_version: '1.2.3.4', run_id: 'run-7', candidate_digest: 'digest', page: '/', width: 1280 })
    expect(bundle.blocked_requests).toEqual(['https://beacon.example/x'])
    expect(bundle.elements).toEqual([sampleElement])
  })

  it('the ported validator refuses the first missing field, a wrong schema or width, and tolerates extra metadata', () => {
    const good = JSON.parse(JSON.stringify(kit.buildBundle(bundleInput)))
    expect(judgeValidate(good)).toBeNull()
    expect(judgeValidate({ ...good, surprise: 1 })).toBeNull()
    expect(judgeValidate({ ...good, schema: 'v2' })).toMatch(/^unknown bundle schema/)
    expect(judgeValidate({ ...good, width: 1024 })).toBe('bundle width 1024 is not one of 1280,375')
    const { elements: _dropped, ...noElements } = good
    expect(judgeValidate(noElements)).toBe('bundle is missing field elements')

    for (const field of ELEMENT_FIELDS) {
      const { [field]: _gone, ...rest } = sampleElement as Record<string, unknown>
      expect(judgeValidate({ ...good, elements: [rest] }), field).toBe(`element 0 is missing field ${field}`)
    }

    const { 'font-face': _face, ...style } = sampleElement.style
    expect(judgeValidate({ ...good, elements: [{ ...sampleElement, style }] })).toBe('element 0 style is missing field font-face')
    const { height: _h, ...box } = sampleElement.box
    expect(judgeValidate({ ...good, elements: [{ ...sampleElement, box }] })).toBe('element 0 box is missing field height')
  })

  it('refuses the old self-authored element shape with the reviewer\'s message, and catches mistyped values and stray element fields', () => {
    const old = { key: 'h1', tag: 'h1', text: 'hi', pinned: false, box: sampleElement.box, type: {}, colour: {} }
    const good = JSON.parse(JSON.stringify(kit.buildBundle(bundleInput)))
    expect(contractProblems({ ...good, elements: [old] })).toEqual(['element 0 is missing field landmarks'])
    const bad = { ...sampleElement, fixed: 'no', parent: 'x', landmarks: ['header'], style: { ...sampleElement.style, 'line-height': '1.5', color: [1, 2, 3] }, key: 'k' }
    expect(contractProblems({ ...good, elements: [bad] })).toEqual(expect.arrayContaining([
      'element 0 fixed is a boolean', 'element 0 parent is an identity or null', 'element 0 landmarks are [tag, heading] pairs',
      'element 0 line-height is a number or normal', 'element 0 color is null or [r, g, b, a]', 'element 0.key: unexpected',
    ]))
    const { run_id: _run, ...noRun } = good
    expect(contractProblems(noRun)).toEqual(['bundle is missing metadata run_id'])
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

  it('on the prototype side only the prototype document and inline resources are allowed', () => {
    const only = '/__prototype__/design.html'

    for (const url of [`${origin}${only}`, `${origin}${only}?x=1`, 'data:image/png;base64,AA==', 'blob:http://127.0.0.1:5000/1', 'about:srcdoc']) {
      expect(kit.classifyRequest(url, origin, only), url).toBe('allowed')
    }

    for (const url of [`${origin}/`, `${origin}/missing.css`, `${origin}/__prototype__/other.css`, 'https://fonts.example/f.woff2']) {
      expect(kit.classifyRequest(url, origin, only), url).toBe('blocked')
    }
  })

  it('fails closed with the exact sentence when a prototype asked for anything outside itself', () => {
    expect(() => kit.assertSelfContained([])).not.toThrow()
    expect(() => kit.assertSelfContained(['https://fonts.example/f.woff2'])).toThrow(/^the chosen design file is not self-contained$/)
  })

  it('finds closed shadow roots anywhere in a pierced DOM tree, including frame documents, and ignores the rest', () => {
    const node = (nodeName: string, extra: Record<string, unknown> = {}) => ({ nodeName, ...extra })
    const root = (shadowRootType: string) => node('#document-fragment', { shadowRootType, children: [node('H2')] })
    const page = (...kids: unknown[]) => node('#document', { children: [node('HTML', { children: kids })] })

    expect(kit.findClosedShadowRoot(page(node('DIV', { shadowRoots: [root('open'), root('user-agent')] })))).toBeNull()
    expect(kit.findClosedShadowRoot(page(node('TEMPLATE', { templateContent: page(node('P', { shadowRoots: [root('closed')] })) })))).toBeNull()
    expect(kit.findClosedShadowRoot(page(node('SECTION', { shadowRoots: [root('closed')] })))).toBe('SECTION')
    expect(kit.findClosedShadowRoot(page(node('IFRAME', { contentDocument: page(node('X-CARD', { shadowRoots: [root('open'), root('closed')] })) })))).toBe('X-CARD')
    expect(kit.findClosedShadowRoot(page(node('DIV', { shadowRoots: [{ ...root('open'), children: [node('ASIDE', { shadowRoots: [root('closed')] })] }] })))).toBe('ASIDE')
  })

  it('forces the rendering values from a cascade layer that is declared first, so it also beats ID-selector important rules', () => {
    expect(kit.STYLE.startsWith('@layer ')).toBe(true)

    for (const [name, value] of Object.entries(kit.FORCED)) {
      expect(kit.STYLE, name).toContain(`${name}:${value}!important`)
    }

    expect(kit.FORCED).toMatchObject({ '-webkit-font-smoothing': 'antialiased', 'text-rendering': 'geometricPrecision', animation: 'none' })
  })

  it('names outputs with a readable slug plus a digest of the exact route', () => {
    const digest = (route: string) => createHash('sha256').update(route).digest('hex').slice(0, 8)
    expect(kit.pageSlug('/a-b')).toBe(`a-b-${digest('/a-b')}`)
    expect(kit.pageSlug('/a/b')).toBe(`a-b-${digest('/a/b')}`)
    expect(kit.pageSlug('/a-b')).not.toBe(kit.pageSlug('/a/b'))
    expect(kit.pageSlug('/')).toBe(`index-${digest('/')}`)
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

  it('ships a valid pin and refuses a binary with a different hash', () => {
    expect(kit.CHROMIUM_PIN.version).toBe('153.0.8010.0')
    expect(kit.CHROMIUM_PIN.package).toBe('@sparticuz/chromium@153.0.0')
    expect(kit.CHROMIUM_PIN.sha256).toMatch(/^[0-9a-f]{64}$/)
    expect(kit.CHROMIUM_PIN.sha256).toBe('53a15d6c3a3d27dfb54c4ba60278b1683136f70cf1e67e989da7dfbd3d451ef0')
    expect(() => kit.verifyChromium(binary, kit.CHROMIUM_PIN, deps)).toThrow('does not match the pin')
  })

  it('the shipped CLI gets past the pin check and refuses the wrong binary by hash', () => {
    const run = spawnSync(process.execPath, [captureFile, '--side', 'candidate', '--pages', '/', '--out', join(scratch, 'cli-out'),
      '--run-id', 'r', '--candidate-digest', 'd', '--chromium', binary], { encoding: 'utf8' })

    expect(run.status).toBe(1)
    expect(run.stderr).toContain('does not match the pin')
    expect(run.stderr).not.toContain('not pinned')
  })
})

// PNG decoding for the stitched-image regressions (8-bit RGB or RGBA, as Chromium's canvas writes).
function decodePng(buf: Buffer) {
  const idat: Buffer[] = []
  let w = 0, h = 0, bpp = 4

  for (let p = 8; p < buf.length;) {
    const len = buf.readUInt32BE(p), type = buf.toString('latin1', p + 4, p + 8), data = buf.subarray(p + 8, p + 8 + len)

    if (type === 'IHDR') { w = data.readUInt32BE(0); h = data.readUInt32BE(4); bpp = data[9] === 6 ? 4 : 3 }

    if (type === 'IDAT') {idat.push(data)}

    p += 12 + len
  }

  const raw = inflateSync(Buffer.concat(idat)), stride = w * bpp, px = Buffer.alloc(h * stride)

  for (let y = 0; y < h; y++) {
    const filter = raw[y * (stride + 1)]

    for (let x = 0; x < stride; x++) {
      const a = x >= bpp ? px[y * stride + x - bpp] : 0, b = y ? px[(y - 1) * stride + x] : 0
      const c = x >= bpp && y ? px[(y - 1) * stride + x - bpp] : 0
      const pa = Math.abs(b - c), pb = Math.abs(a - c), pc = Math.abs(a + b - 2 * c)
      const paeth = pa <= pb && pa <= pc ? a : pb <= pc ? b : c
      px[y * stride + x] = (raw[y * (stride + 1) + 1 + x] + [0, a, b, (a + b) >> 1, paeth][filter]) & 255
    }
  }

  return { w, h, at: (x: number, y: number) => [...px.subarray(y * stride + x * bpp, y * stride + x * bpp + 3)] }
}

describe.skipIf(process.env.DESIGN_COMPARE_REAL_CHROMIUM !== '1')('real Chromium (opt-in, pinned binary from CHROME_PATH, shipped pin)', () => {
  const chromium = process.env.CHROME_PATH as string
  let seq = 0

  const dirOf = (files: Record<string, string>) => {
    const d = join(scratch, `case-${++seq}`)

    for (const [name, body] of Object.entries(files)) {
      mkdirSync(join(d, name, '..'), { recursive: true })
      writeFileSync(join(d, name), body)
    }

    return d
  }

  const shoot = (site: Record<string, string>, pages = ['/']) => {
    const outDir = join(scratch, `out-${++seq}`)

    return kit.capture({ side: 'candidate', siteDir: dirOf(site), pages, outDir, chromium, runId: 'run-1', candidateDigest: 'd1' })
      .then((bundles: any[]) => ({ bundles, outDir }))
  }

  const sameFiles = (a: string, b: string, route = '/') => {
    for (const file of ['1280.json', '375.json', '1280.png', '375.png']) {
      const name = `${kit.pageSlug(route)}-${file}`
      expect(readFileSync(join(a, name)).equals(readFileSync(join(b, name))), name).toBe(true)
    }
  }

  const receiver = async () => {
    const hits: string[] = []
    const server = createServer((req, res) => { hits.push(req.url as string); res.end('x') })
    server.on('upgrade', (req, socket) => { hits.push(`upgrade ${req.url}`); socket.destroy() })
    await new Promise<void>(r => server.listen(0, '127.0.0.1', r))

    return { base: `127.0.0.1:${(server.address() as { port: number }).port}`, hits, close: () => new Promise(r => server.close(r)) }
  }

  const motionPage = (ms: number, smoothing: string) => `<!doctype html><meta charset=utf-8>
<style>body{margin:0;font:20px sans-serif;-webkit-font-smoothing:${smoothing}}
@keyframes spin{to{transform:rotate(360deg)}}
#box{width:80px;height:80px;background:#369;animation:spin ${ms}ms linear infinite}
#wa{width:80px;height:60px;background:red}h1{margin:16px}</style><h1>Tiny page</h1><div id=box></div><div id=wa></div><p>Some text</p>
<div style="opacity:0"><p>Invisible text</p></div>
<div style="visibility:hidden"><p style="visibility:visible">Shown override</p></div><p>Visible control</p>
<header><h1>Site title</h1><nav><a href="/x" aria-label="Go home">Home</a></nav></header>
<main><section role="region"><h2>Part one</h2><p style="font:italic 700 18px/30px Georgia,serif;color:rgba(10,20,30,.5)">Body text</p>
<img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" alt="A dot" width=20 height=20><button>Go</button></section></main>
<div style="position:fixed;bottom:0;left:0;color:#369">Pinned</div>
<script>document.getElementById('wa').animate([{transform:'translateX(0px)'},{transform:'translateX(300px)'}],{duration:${ms},iterations:Infinity})</script>`

  let motionRuns: Promise<[Awaited<ReturnType<typeof shoot>>, Awaited<ReturnType<typeof shoot>>]> | undefined
  const motion = () => (motionRuns ??= Promise.all([shoot({ 'index.html': motionPage(1000, 'antialiased') }), shoot({ 'index.html': motionPage(3700, 'subpixel-antialiased') })]))

  it('gives identical bundles and images despite CSS animation, infinite Web Animations API motion and font smoothing', async () => {
    const [a, b] = await motion()
    sameFiles(a.outDir, b.outDir)
    const bundle = JSON.parse(readFileSync(join(a.outDir, `${kit.pageSlug('/')}-1280.json`), 'utf8'))
    expect(contractProblems(bundle)).toEqual([])
    expect(bundle.elements.find((e: any) => e.box.width === 80 && e.box.height === 60).box.x).toBe(0)
    expect(bundle.chromium_version).toBe('153.0.8010.0')
  }, 120_000)

  it('emits bundles that satisfy the C1 contract at both widths: landmarks, role, label, alt, parent, fixed and the STYLE_FIELDS names', async () => {
    const [{ bundles }] = await motion()
    expect(bundles.map((b: any) => b.width)).toEqual([1280, 375])

    for (const bundle of bundles) {
      expect(contractProblems(JSON.parse(JSON.stringify(bundle))), `${bundle.width}`).toEqual([])
      const keys = judgeKeys(bundle.elements)
      expect(keys).toEqual(expect.arrayContaining([
        'header:site title/h1:"site title"#1', 'header:site title/nav/a:"go home"#1', 'main:part one/region:""#1',
        'main:part one/section:part one/img:"a dot"#1', 'main:part one/section:part one/button:"go"#1', '/div:"pinned"#1',
      ]))
      expect(new Set(keys).size).toBe(keys.length)
      const body = byText(bundle, 'body text')
      expect(body.style).toMatchObject({ 'font-size': 18, 'font-weight': 700, 'line-height': 30, color: [10, 20, 30, 0.5] })
      expect(body.style['font-family']).toBe('Georgia, serif')
      expect(body.landmarks).toEqual([['main', 'part one'], ['section', 'part one']])
      expect(byText(bundle, 'pinned').fixed).toBe(true)
      expect(body.fixed).toBe(false)
      expect(Number.isInteger(body.parent)).toBe(true)
      expect(bundle.elements.filter((e: any) => e.parent === body.parent).map((e: any) => e.tag)).toEqual(['h2', 'p', 'img', 'button'])
      expect(bundle.run_id).toBe('run-1')
      expect(bundle.candidate_digest).toBe('d1')
      expect(bundle.kit_sha256).toMatch(/^[0-9a-f]{64}$/)
    }
  }, 120_000)

  it('excludes elements under an opacity-zero ancestor and keeps visible controls', async () => {
    const keys = judgeKeys((await motion())[0].bundles[0].elements)
    expect(keys).toContain('/p:"visible control"#1')
    expect(keys).toContain('/p:"shown override"#1')
    expect(keys.join('\n')).not.toContain('invisible text')
  }, 120_000)

  it('refuses outside WebSocket, EventSource, beacon, workers and fetch before they reach a separate receiver', async () => {
    const r = await receiver()

    const page = `<!doctype html><h1>net</h1><script>
for (const f of [() => new WebSocket('ws://${r.base}/socket'), () => new EventSource('http://${r.base}/es'),
  () => navigator.sendBeacon('http://${r.base}/beacon', 'x'), () => new Worker('http://${r.base}/w.js'), () => new Worker('/w.js'),
  () => new SharedWorker('/w.js'), () => navigator.serviceWorker.register('/w.js'),
  () => fetch('http://${r.base}/fetch').catch(() => {})]) { try { f() } catch {} }</script>`

    const { bundles } = await shoot({ 'index.html': page, 'w.js': `new WebSocket('ws://${r.base}/from-worker')` })
    const wanted = [`ws://${r.base}/socket`, `http://${r.base}/es`, `http://${r.base}/beacon`, `http://${r.base}/w.js`, `http://${r.base}/fetch`]

    for (const b of bundles) {
      expect(b.blocked_requests).toEqual(expect.arrayContaining(wanted))
      expect(b.blocked_requests.some((u: string) => u.endsWith('/w.js') && !u.includes(r.base))).toBe(true)
    }

    const proto = join(scratch, 'net-proto.html')
    writeFileSync(proto, page)
    await expect(kit.capture({ side: 'prototype', prototypeFile: proto, pages: ['/'], outDir: join(scratch, 'net-proto-out'), chromium, runId: 'r', candidateDigest: 'd' }))
      .rejects.toThrow(/^the chosen design file is not self-contained$/)
    await r.close()
    expect(r.hits).toEqual([])
  }, 120_000)

  it('stops a prototype that references same-origin or outside resources, and accepts an inline one', async () => {
    const r = await receiver()
    const site = dirOf({ 'missing-required.css': 'h1{color:red}', 'asset.css': 'h1{color:red}' })
    const file = join(scratch, 'proto-leaks.html')
    writeFileSync(file, `<link rel=stylesheet href="asset.css"><link rel=stylesheet href="/missing-required.css">
<link rel=stylesheet href="http://${r.base}/x.css"><h1>P</h1>`)

    const err: any = await kit.capture({ side: 'prototype', siteDir: site, prototypeFile: file, pages: ['/'], outDir: join(scratch, 'proto-out-leaks'), chromium, runId: 'r', candidateDigest: 'd' })
      .catch((e: unknown) => e)

    expect(err.message).toBe('the chosen design file is not self-contained')

    for (const suffix of ['/__prototype__/asset.css', '/missing-required.css', `${r.base}/x.css`]) {
      expect(err.blocked.some((u: string) => u.endsWith(suffix)), suffix).toBe(true)
    }

    await r.close()
    expect(r.hits).toEqual([])
    const ok = join(scratch, 'proto-ok.html')
    writeFileSync(ok, '<style>h1{color:#369}</style><h1>P</h1><img alt=dot src="data:image/gif;base64,R0lGODlhAQABAAAAACw=">')
    const bundles = await kit.capture({ side: 'prototype', siteDir: site, prototypeFile: ok, pages: ['/'], outDir: join(scratch, 'proto-out-ok'), chromium, runId: 'r', candidateDigest: 'd' })
    expect(bundles.map((b: any) => b.blocked_requests)).toEqual([[], []])
  }, 120_000)

  it('keeps first-tile pixels when the last tile overlaps and fixed or sticky elements are hidden', async () => {
    const stripe = (y: number) => (Math.floor(y / 50) % 2 ? [51, 153, 204] : [204, 51, 51])

    const page = (height: number) => `<!doctype html><style>html,body{margin:0}
#s{box-sizing:border-box;padding-top:300px;height:${height}px;background:repeating-linear-gradient(#cc3333 0 50px,#3399cc 50px 100px)}
#k{position:sticky;top:0;width:200px;height:40px;background:#00aa00}
#h{position:fixed;top:0;left:0;width:80px;height:60px;background:#ff0000}</style><div id=s><div id=k></div></div><div id=h></div>`

    const { outDir } = await shoot({ 'a.html': page(836), 'b.html': page(1750) }, ['/a', '/b'])

    for (const [route, height] of [['/a', 836], ['/b', 1750]] as const) {
      for (const width of [1280, 375]) {
        const png = decodePng(readFileSync(join(outDir, `${kit.pageSlug(route)}-${width}.png`)))
        expect(png.h, `${route} ${width} height`).toBe(height)

        for (let y = 0; y < height; y++) {
          const label = `${route} ${width} y=${y}`
          expect(png.at(300, y), label).toEqual(stripe(y))
          expect(png.at(40, y), `${label} x=40`).toEqual(y < 60 ? [255, 0, 0] : y >= 300 && y < 340 ? [0, 170, 0] : stripe(y))
        }

        expect(png.at(10, 59), `${route} ${width} header bottom`).toEqual([255, 0, 0])
      }
    }
  }, 120_000)

  it('writes one bundle and PNG per route and width through the shipped CLI, never overwriting', async () => {
    const site = dirOf({ 'a-b.html': '<h1>Dash route</h1>', 'a/b.html': '<h1>Slash route</h1>' })
    const out = join(scratch, 'cli-e2e')

    const cli = (pages: string, outDir = out) => spawnSync(process.execPath, [captureFile, '--side', 'candidate', '--site', site, '--pages', pages,
      '--out', outDir, '--run-id', 'r', '--candidate-digest', 'd'], { encoding: 'utf8', env: { ...process.env, CHROME_PATH: chromium } })

    const run = cli('/a-b /a/b')
    expect(run.status, run.stderr).toBe(0)
    const files = readdirSync(out)
    expect(files.filter(f => f.endsWith('.json'))).toHaveLength(4)
    expect(files.filter(f => f.endsWith('.png'))).toHaveLength(4)

    for (const [route, heading] of [['/a-b', 'dash route'], ['/a/b', 'slash route']]) {
      for (const width of [1280, 375]) {
        const bundle = JSON.parse(readFileSync(join(out, `${kit.pageSlug(route)}-${width}.json`), 'utf8'))
        expect(bundle.page).toBe(route)
        expect(bundle.image).toBe(`${kit.pageSlug(route)}-${width}.png`)
        expect(bundle.elements.map((e: any) => e.text)).toContain(heading)
        expect(files).toContain(bundle.image)
      }
    }

    const again = cli('/a-b')
    expect(again.status).toBe(1)
    expect(again.stderr).toContain('refusing to overwrite')
    expect(readdirSync(out)).toHaveLength(8)
    const dup = cli('/a-b /a-b', join(scratch, 'cli-dup'))
    expect(dup.status).toBe(1)
    expect(dup.stderr).toContain('refusing to overwrite')
  }, 120_000)

  const settleHits = () => new Promise(r => setTimeout(r, 500))

  const asPrototype = (html: string) => {
    const file = join(scratch, `proto-${++seq}.html`)
    writeFileSync(file, html)

    return kit.capture({ side: 'prototype', prototypeFile: file, pages: ['/'], outDir: join(scratch, `out-${++seq}`), chromium, runId: 'r', candidateDigest: 'd' })
      .catch((e: unknown) => e)
  }

  const read = (outDir: string, width = 1280) => JSON.parse(readFileSync(join(outDir, `${kit.pageSlug('/')}-${width}.json`), 'utf8'))
  const byText = (bundle: any, text: string) => bundle.elements.find((e: any) => e.text === text)

  const escapes: [string, (base: string) => string][] = [
    ['direct popup navigation', base => `<h1>popup</h1><script>window.open('http://${base}/popup')</script>`],
    ['an about:blank popup native WebSocket', base => `<h1>realm</h1><script>const p = window.open('about:blank'); new p.WebSocket('ws://${base}/clean-socket')</script>`],
  ]

  for (const [name, page] of escapes) {
    it(`never lets ${name} reach a separate receiver, on the candidate and on the prototype side`, async () => {
      const r = await receiver()
      const { bundles } = await shoot({ 'index.html': page(r.base) })

      for (const b of bundles) {expect(b.blocked_requests.length).toBeGreaterThan(0)}

      const err: any = await asPrototype(page(r.base))
      expect(err.message).toBe('the chosen design file is not self-contained')
      expect(err.blocked.length).toBeGreaterThan(0)
      await settleHits()
      await r.close()
      expect(r.hits).toEqual([])
    }, 120_000)
  }

  it('also keeps link, form, frame-realm and cross-site frame routes from reaching a separate receiver', async () => {
    const r = await receiver()
    const other = r.base.replace('127.0.0.1', 'localhost')

    const routes = `<h1>routes</h1><a id=a target=_blank href="http://${r.base}/link">l</a><form id=f target=_blank action="http://${r.base}/form"></form><script>
const frame = () => document.body.appendChild(document.createElement('iframe'))
for (const f of [() => a.click(), () => f.submit(), () => new (frame().contentWindow.WebSocket)('ws://${r.base}/frame-socket'),
  () => frame().contentWindow.open('http://${r.base}/frame-popup'), () => { frame().src = 'http://${other}/frame' }]) { try { f() } catch {} }</script>`

    // The blocked cross-site frame cannot be read, so the capture itself is refused; what matters is that nothing reached the receiver.
    await shoot({ 'index.html': routes }).catch(() => {})
    await settleHits()
    await r.close()
    expect(r.hits).toEqual([])
  }, 120_000)

  const shadowPage = (ms: number) => `<!doctype html><h1>Outside</h1><div id=host></div><script>host.attachShadow({mode:'open'}).innerHTML =
'<style>@keyframes m{to{transform:translateX(300px)}}h2{width:100px;animation:m ${ms}ms infinite linear}</style><h2>Inside</h2>'</script>`

  it('gives identical bundles and pixels for a shadow-root element with an infinite CSS animation when only its period differs', async () => {
    const [a, b] = await Promise.all([shoot({ 'index.html': shadowPage(1000) }), shoot({ 'index.html': shadowPage(10_000) })])
    sameFiles(a.outDir, b.outDir)
    expect(byText(read(a.outDir), 'inside').tag).toBe('h2')
  }, 120_000)

  it('measures visible shadow-root and srcdoc-iframe text with its own type values and page coordinates', async () => {
    const shadow = (px: number) => `<h1>Outside</h1><div id=host></div><script>host.attachShadow({mode:'open'}).innerHTML = '<p style="font-size:${px}px">Visible shadow text</p><canvas width=50 height=20></canvas>'</script>`
    const frame = (px: number) => `<h1>Outside</h1><iframe srcdoc="<h2 style='font-size:${px}px'>Frame text</h2><p style='position:fixed;bottom:0'>Pin</p>" style="border:0;margin:30px"></iframe>`

    const runs = await Promise.all([shadow(10), shadow(40), frame(10), frame(40)].map(html => shoot({ 'index.html': html })))
    const [s10, s40, f10, f40] = runs.map(x => read(x.outDir))

    expect(byText(s10, 'visible shadow text').style['font-size']).toBe(10)
    expect(byText(s40, 'visible shadow text').style['font-size']).toBe(40)
    expect(byText(f10, 'frame text').style['font-size']).toBe(10)
    expect(byText(f40, 'frame text').style['font-size']).toBe(40)
    expect(JSON.stringify(s10)).not.toBe(JSON.stringify(s40))
    expect(JSON.stringify(f10)).not.toBe(JSON.stringify(f40))
    expect(byText(f10, 'frame text').box.x).toBeGreaterThanOrEqual(40)
    expect(byText(f10, 'frame text').box.y).toBeGreaterThan(60)
    expect(s10.masks).toContainEqual(expect.objectContaining({ width: 50, height: 20 }))
    expect(byText(f10, 'pin').fixed).toBe(true)
  }, 120_000)

  it('refuses closed shadow roots, declarative closed roots and unreadable frames instead of claiming complete evidence', async () => {
    const pages = {
      closed: '<div id=h></div><script>h.attachShadow({mode:"closed"}).innerHTML = "<h2>Hidden</h2>"</script>',
      declarative: '<div><template shadowrootmode="closed"><h2>Hidden</h2></template></div>',
      sandboxed: '<iframe sandbox srcdoc="<h2>Opaque</h2>"></iframe>',
    }

    for (const [name, html] of Object.entries(pages)) {
      const err: any = await shoot({ 'index.html': html }).catch((e: unknown) => e)
      expect(err.message, name).toMatch(name === 'sandboxed' ? /frame/ : /closed shadow root/)
    }
  }, 120_000)

  const stunTurn = async () => {
    const seen = { packets: 0, connections: 0 }
    const udp = createSocket('udp4')
    udp.on('message', () => { seen.packets++ })
    await new Promise<void>(r => udp.bind(0, '127.0.0.1', r))
    const tcp = createTcpServer(socket => { seen.connections++; socket.destroy() })
    await new Promise<void>(r => tcp.listen(0, '127.0.0.1', r))
    const stun = `stun:127.0.0.1:${udp.address().port}`
    const turn = `turn:127.0.0.1:${(tcp.address() as { port: number }).port}?transport=tcp`

    return { seen, stun, turn, close: () => Promise.all([new Promise<void>(r => udp.close(() => r())), new Promise<void>(r => tcp.close(() => r()))]) }
  }

  it('refuses WebRTC, WebTransport and data channels in the page, srcdoc and script-made frames before ICE reaches a STUN or TURN receiver, on both sides', async () => {
    const r = await stunTurn()
    const ice = `{iceServers:[{urls:'${r.stun}'},{urls:'${r.turn}',username:'u',credential:'c'}]}`

    const run = `for (const C of [RTCPeerConnection, webkitRTCPeerConnection]) { try { const pc = new C(${ice}); pc.createDataChannel('x'); pc.createOffer().then(o => pc.setLocalDescription(o)).catch(() => {}) } catch {} }
try { new WebTransport('https://127.0.0.1:9/') } catch {}`

    const page = `<!doctype html><h1>rtc</h1><script>${run}</script><iframe srcdoc="<h2>frame</h2><script>${run}</script>"></iframe>
<script>const f = document.body.appendChild(document.createElement('iframe')); try { new f.contentWindow.RTCPeerConnection(${ice}) } catch {}</script>`

    const { bundles } = await shoot({ 'index.html': page })

    for (const b of bundles) {expect(b.blocked_requests).toEqual(expect.arrayContaining([r.stun, r.turn, 'webrtc:RTCPeerConnection', 'https://127.0.0.1:9/']))}

    const err: any = await asPrototype(page)
    expect(err.message).toBe('the chosen design file is not self-contained')
    expect(err.blocked).toEqual(expect.arrayContaining([r.stun, r.turn]))
    await new Promise(done => setTimeout(done, 1000))
    await r.close()
    expect(r.seen).toEqual({ packets: 0, connections: 0 })
  }, 120_000)

  const bodyShadowPage = (ms: number) => `<!doctype html><body><script>document.body.attachShadow({mode:'open'}).innerHTML =
'<style>@keyframes m{to{transform:translateX(300px)}}h2{width:100px;animation:m ${ms}ms infinite linear}</style><h2>Inside body</h2>'; document.currentScript.remove()</script>`

  it('visits a body-attached open shadow root with an animated h2 and gives period-independent images', async () => {
    const [a, b] = await Promise.all([shoot({ 'index.html': bodyShadowPage(1000) }), shoot({ 'index.html': bodyShadowPage(10_000) })])
    sameFiles(a.outDir, b.outDir)

    for (const width of [1280, 375]) {expect(byText(read(a.outDir, width), 'inside body').tag).toBe('h2')}
  }, 120_000)

  it('normalises before page code reads it and measures each body: one synchronous computed-style read of inline !important, a shadow root, a declarative shadow root and a srcdoc frame', async () => {
    const page = (smoothing: string, rendering: string) => {
      const decl = `-webkit-font-smoothing:${smoothing};text-rendering:${rendering}`
      const reflect = (id: string, out: string) => `${out}.textContent = getComputedStyle(${id}).webkitFontSmoothing + ' / ' + getComputedStyle(${id}).textRendering`
      const frame = `Frame body text<p id=a style='${decl}!important'>A</p><p id=o></p><script>${reflect('a', 'o')}</script>`.replace(/&/g, '&amp;').replace(/"/g, '&quot;')

      return `<!doctype html><body style="background:#eee">Top body text<p id=a style="${decl}!important">Inline</p><p id=ra></p><script>${reflect('a', 'ra')}</script>
<div id=host></div><script>const r = host.attachShadow({mode:'open'}), own = new CSSStyleSheet(); own.replaceSync('h2{${decl}}')
r.adoptedStyleSheets = [own]; r.innerHTML = '<h2 id=s>S</h2><p id=o></p>'
const o = r.getElementById('o'); o.textContent = getComputedStyle(r.getElementById('s')).webkitFontSmoothing + ' / ' + getComputedStyle(r.getElementById('s')).textRendering</script>
<div id=dh><template shadowrootmode=open><style>h3{${decl}}</style><h3 id=d>D</h3><p id=e></p></template></div>
<script>const dr = dh.shadowRoot; dr.getElementById('e').textContent = getComputedStyle(dr.getElementById('d')).webkitFontSmoothing + ' / ' + getComputedStyle(dr.getElementById('d')).textRendering</script>
<iframe srcdoc="${frame}"></iframe>`
    }

    const [a, b] = await Promise.all([shoot({ 'index.html': page('subpixel-antialiased', 'optimizeSpeed') }), shoot({ 'index.html': page('none', 'optimizeLegibility') })])

    for (const width of [1280, 375]) {
      const shown = read(a.outDir, width).elements.filter((e: any) => e.text === 'antialiased / geometricprecision')
      expect(shown, `${width}`).toHaveLength(4)
      expect(JSON.stringify(read(b.outDir, width))).toBe(JSON.stringify(read(a.outDir, width)))
      const bodies = read(a.outDir, width).elements.filter((e: any) => e.tag === 'body')
      expect(bodies.map((e: any) => e.text).sort(), `${width} body text`).toEqual(['frame body text', 'top body text'])
      expect(bodies.find((e: any) => e.text === 'top body text').style['background-color']).toEqual([238, 238, 238, 1])
    }
  }, 120_000)

  it('enforces the rendering values against ID-selector declarations and inline !important set again by the page, and fails when the page empties the layer', async () => {
    const forced = '-webkit-font-smoothing:subpixel-antialiased!important;text-rendering:optimizeSpeed!important'

    const page = (script: string) => `<style>#a{${forced}}</style><h1 id=a>AVAVATAR</h1><p id=b style="${forced}">Inline</p><p id=ra></p><p id=rb></p>
<script>const show = () => { for (const [s, d] of [[a, ra], [b, rb]]) { const c = getComputedStyle(s); d.textContent = c.webkitFontSmoothing + ' / ' + c.textRendering } ${script} requestAnimationFrame(show) }; show()</script>`

    const { outDir } = await shoot({ 'index.html': page(`b.style.setProperty('-webkit-font-smoothing', 'subpixel-antialiased', 'important');`) })

    for (const width of [1280, 375]) {
      const shown = read(outDir, width).elements.filter((e: any) => e.text === 'antialiased / geometricprecision')
      expect(shown, `${width}`).toHaveLength(2)
    }

    const err: any = await shoot({ 'index.html': `${page('')}<script>addEventListener('load', () => document.getElementById('__dc_style').textContent = '')</script>` }).catch((e: unknown) => e)
    expect(err.message).toContain('rendering normalisation is not effective')
  }, 120_000)

  const fontPage = (face: string, used: string) => `<!doctype html><style>
@font-face{font-family:${face};src:url(/acme.woff2) format('woff2')}
body{margin:0;font-size:32px}#t{display:inline-block;font-family:${used};margin:0}</style><p id=t>Measured text</p>`

  const fontSite = (face: string, used: string) => {
    const site = dirOf({ 'index.html': fontPage(face, used) })
    copyFileSync(fileURLToPath(new URL('../web/public/fonts/Mondwest-Regular.woff2', import.meta.url)), join(site, 'acme.woff2'))

    return kit.capture({ side: 'candidate', siteDir: site, pages: ['/'], outDir: join(scratch, `out-${++seq}`), chromium, runId: 'r', candidateDigest: 'd' })
  }

  it('keeps comma-containing quoted font families apart on the measured element, and equal when the names match', async () => {
    const [same, other, matching] = await Promise.all([
      fontSite('"Acme,Display"', '"Acme,Display", serif'), fontSite('"Acme,Display"', '"Acme,Other", serif'), fontSite("'Acme,Display'", "'ACME,display', monospace"),
    ])

    for (let i = 0; i < 2; i++) {
      const [a, b, c] = [same, other, matching].map(r => byText(r[i], 'measured text'))
      const first = (e: any) => probe.firstFamily(e.style['font-family'])
      expect(a.style['font-family'], `${i}`).toBe('"Acme,Display", serif')
      expect(b.style['font-family']).toBe('"Acme,Other", serif')
      expect([first(a), first(b)]).toEqual(['acme,display', 'acme,other'])
      expect(a.style['font-face']).toBe('acme,display')
      expect(b.style['font-face']).toBe('')
      expect(a.box.width).not.toBe(b.box.width)
      expect(first(c)).toBe(first(a))
      expect(c.style['font-face']).toBe(first(a))
      expect(c.box).toEqual(a.box)
      expect(c.style['font-face']).toBe(a.style['font-face'])
    }
  }, 120_000)

  const dynamic = (smoothing: string, rendering: string) => {
    const decl = `-webkit-font-smoothing:${smoothing};text-rendering:${rendering}`

    return `(() => {
  const show = (root, p) => { const c = getComputedStyle(p); const q = document.createElement('p'); q.textContent = c.webkitFontSmoothing + ' / ' + c.textRendering; root.appendChild(q) }
  const make = root => {
    let p = document.createElement('p'); p.textContent = 'one'
    p.style.setProperty('-webkit-font-smoothing', '${smoothing}', 'important'); p.style.setProperty('text-rendering', '${rendering}', 'important')
    root.appendChild(p); show(root, p)
    p = document.createElement('p'); p.textContent = 'two'; p.style.cssText = '${decl}!important'; root.appendChild(p); show(root, p)
    p = document.createElement('p'); p.textContent = 'three'; p.setAttribute('style', '${decl}!important'); root.appendChild(p); show(root, p)
    p = document.createElement('p'); p.textContent = 'four'; p.style = '${decl}!important'; root.appendChild(p); show(root, p)
    const host = document.createElement('div'); root.appendChild(host); host.innerHTML = '<span style="${decl}!important">five</span>'; show(root, host.firstChild)
    p = document.createElement('p'); p.textContent = 'six'; const live = getComputedStyle(p); root.appendChild(p)
    p.style.setProperty('-webkit-font-smoothing', '${smoothing}', 'important'); p.style.setProperty('text-rendering', '${rendering}', 'important')
    const q = document.createElement('p'); q.textContent = live.webkitFontSmoothing + ' / ' + live.textRendering; root.appendChild(q)
  }
  return make
})()`
  }

  const dynamicPage = (smoothing: string, rendering: string) => {
    const make = dynamic(smoothing, rendering)
    const frame = `<body>Frame text<script>${make}(document.body)</script>`.replace(/&/g, '&amp;').replace(/"/g, '&quot;')

    return `<!doctype html><body>Top text<div id=top></div><div id=host></div><script>
${make}(document.getElementById('top'))
host.attachShadow({mode:'open'}); ${make}(host.shadowRoot)</script><iframe srcdoc="${frame}"></iframe>`
  }

  it('normalises synchronously: created, style-set (setProperty, cssText, style attribute, style property), inserted and read-back elements in the page, an open shadow root and a srcdoc frame', async () => {
    const [a, b] = await Promise.all([
      shoot({ 'index.html': dynamicPage('subpixel-antialiased', 'optimizeSpeed') }), shoot({ 'index.html': dynamicPage('none', 'optimizeLegibility') }),
    ])

    for (const width of [1280, 375]) {
      const shown = read(a.outDir, width).elements.filter((e: any) => e.text === 'antialiased / geometricprecision')
      expect(shown, `${width}`).toHaveLength(18)
      expect(JSON.stringify(read(b.outDir, width))).toBe(JSON.stringify(read(a.outDir, width)))
    }

    sameFiles(a.outDir, b.outDir)
  }, 120_000)

  it('refuses the capture when a retained computed style was read after a mutation path the kit cannot cover, and accepts the same bypass when nothing observed the element', async () => {
    const page = (retain: boolean) => `<!doctype html><p id=a>A</p><p id=o></p><script>
const p = document.getElementById('a'), live = ${retain ? 'getComputedStyle(p)' : 'null'}
p.attributes.style ? 0 : p.setAttribute('style', 'color:red')
p.attributes.style.value = 'text-rendering:optimizeSpeed!important'
document.getElementById('o').textContent = (live || getComputedStyle(p)).textRendering</script>`

    const [err, { bundles }]: any[] = await Promise.all([shoot({ 'index.html': page(true) }).catch((e: unknown) => e), shoot({ 'index.html': page(false) })])
    expect(err.message).toContain('normalisation could not be guaranteed')
    expect(bundles.map((b: any) => b.elements.some((e: any) => e.text === 'geometricprecision'))).toEqual([true, true])
  }, 120_000)
})
