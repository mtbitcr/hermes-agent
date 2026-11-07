import { spawn, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { createSocket } from 'node:dgram'
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { createServer as createTcpServer } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { runInNewContext } from 'node:vm'
import { deflateSync, inflateSync } from 'node:zlib'

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

  it('continues the landmark chain through shadow hosts', () => {
    const inShadow = el('h2', {}, 'Inner')
    const host = el('div', {})
    el('body', {}, el('main', {}, host))
    inShadow.getRootNode = () => ({ host })
    expect(probe.landmarksOf(inShadow)).toEqual([['main', '']])
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

describe('probe canonical scroll state and native motion freeze', () => {
  const win = (x = 0, y = 0, pinned = false) => {
    const w: Record<string, any> = { scrollX: x, scrollY: y }

    w.scrollTo = (o: { left: number, top: number, behavior: string }) => {
      w.last = o

      if (!pinned) { w.scrollX = o.left; w.scrollY = o.top }
    }

    return w
  }

  it('scrolls every window to 0,0 instantly and refuses when one cannot get there', () => {
    const [a, b] = [win(1000, 1000), win(0, 500)]
    probe.scrollToTop([a, b])
    expect([a.scrollX, a.scrollY, b.scrollX, b.scrollY]).toEqual([0, 0, 0, 0])
    expect(a.last).toEqual({ left: 0, top: 0, behavior: 'instant' })
    expect(() => probe.scrollToTop([win(0, 0), win(0, 300, true)])).toThrow('top-of-page scroll state')
  })

  it('asserts the canonical state without moving anything', () => {
    const w = win(0, 0)
    probe.assertTop([w])
    expect(w.last).toBeUndefined()
    expect(() => probe.assertTop([win(0, 0), win(40, 0)])).toThrow('top-of-page scroll state')
    expect(() => probe.assertTop([win(0, 1000)])).toThrow('top-of-page scroll state')
  })

  it('gives fixed and sticky boxes viewport coordinates and every other box document coordinates, without adding the scroll twice', () => {
    const rect = { left: 10.126, top: 20.5, width: 30, height: 40 }
    const ctx = { win: { scrollX: 5, scrollY: 7 } }
    expect(probe.boxOf(rect, ctx, true)).toEqual({ x: 10.13, y: 20.5, width: 30, height: 40 })
    expect(probe.boxOf(rect, ctx, false)).toEqual({ x: 15.13, y: 27.5, width: 30, height: 40 })
  })

  const svgRoot = (opts: { stuck?: boolean } = {}) => {
    const s: Record<string, any> = { localName: 'svg', paused: false, time: 9, calls: [] as string[] }

    s.pauseAnimations = () => { s.calls.push('pause'); s.paused = true }

    s.setCurrentTime = (t: number) => { s.calls.push(`time ${t}`);

 if (!opts.stuck) { s.time = t } }

    s.animationsPaused = () => s.paused
    s.getCurrentTime = () => s.time

    return s
  }

  const marquee = () => {
    const kids = ['text', 'b']
    const fresh: Record<string, any> = { localName: 'marquee', kids: [] as string[], stopped: false }

    fresh.appendChild = (k: string) => { fresh.kids.push(k); kids.shift() }

    fresh.stop = () => { fresh.stopped = true }
    const old: Record<string, any> = { localName: 'marquee', parentNode: {}, replacedBy: null }
    Object.defineProperty(old, 'firstChild', { get: () => kids[0] ?? null })

    old.cloneNode = (deep: boolean) => { old.deep = deep;

 return fresh }

    old.replaceWith = (n: unknown) => { old.replacedBy = n }

    return { old, fresh }
  }

  it('pauses every svg root at time 0 and verifies it, ignoring other elements', () => {
    const [a, b] = [svgRoot(), svgRoot()]
    probe.freezeNative([a, { localName: 'div' }, b])
    expect(a.calls).toEqual(['pause', 'time 0'])
    expect(b.calls).toEqual(['pause', 'time 0'])
    expect(a.time).toBe(0)
    expect(() => probe.freezeNative([svgRoot({ stuck: true })])).toThrow('unsupported content')
  })

  it('replaces a marquee by a stopped, never-started copy that keeps the original children, and refuses a marquee that cannot stop', () => {
    const { old, fresh } = marquee()
    probe.freezeNative([old])
    expect(old.deep).toBe(false)
    expect(old.replacedBy).toBe(fresh)
    expect(fresh.kids).toEqual(['text', 'b'])
    expect(fresh.stopped).toBe(true)
    const broken = marquee()
    delete broken.fresh.stop
    expect(() => probe.freezeNative([broken.old])).toThrow('unsupported content')
  })

  it('is idempotent on repeated assertions and leaves unrelated elements alone', () => {
    const s = svgRoot()
    probe.freezeNative([s, { localName: 'p' }])
    probe.freezeNative([s, { localName: 'p' }])
    expect(s.calls).toEqual(['pause', 'time 0', 'pause', 'time 0'])
  })
})

const REFUSAL = 'unsupported content: the kit captures single documents only'

describe('single documents only', () => {
  const tag = (localName: string, extra: Record<string, unknown> = {}) => ({ localName, ...extra })
  const shadow = (...kids: FakeNode[]) => ({ querySelectorAll: () => kids })

  const probeOf = (...page: FakeNode[]) =>
    runInNewContext(probeSource, { window: {}, document: { createElement: () => ({}), documentElement: {}, body: { querySelectorAll: () => page } } })

  it('refuses an iframe, frame, fencedframe, object or embed element and names it', () => {
    for (const name of ['iframe', 'frame', 'fencedframe', 'object', 'embed']) {
      const page = probeOf(tag('h1'), tag(name))
      expect(() => page.assertSingleDocument(), name).toThrow(`${REFUSAL}: found <${name}>`)
    }
  })

  it('finds the element inside an open shadow root, also a nested one', () => {
    expect(() => probeOf(tag('div', { shadowRoot: shadow(tag('p'), tag('iframe')) })).assertSingleDocument()).toThrow(`${REFUSAL}: found <iframe>`)
    expect(() => probeOf(tag('div', { shadowRoot: shadow(tag('section', { shadowRoot: shadow(tag('embed')) })) })).assertSingleDocument()).toThrow(`${REFUSAL}: found <embed>`)
  })

  it('passes a page without such an element, also with an img, svg, canvas and video', () => {
    const plain = ['html', 'h1', 'p', 'a', 'img', 'svg', 'canvas', 'video', 'input', 'picture', 'source'].map(n => tag(n))
    expect(() => probeOf().assertSingleDocument()).not.toThrow()
    expect(() => probeOf(...plain).assertSingleDocument()).not.toThrow()
    expect(() => probeOf(tag('div', { shadowRoot: shadow(...plain) })).assertSingleDocument()).not.toThrow()
  })

  it('refuses a page whose frame tree holds a child frame, by url or by id, and passes a lone main frame', () => {
    const frame = (id: string, url = '') => ({ id, url })
    expect(() => kit.assertNoChildFrames({ frame: frame('main', 'http://x/') })).not.toThrow()
    expect(() => kit.assertNoChildFrames({ frame: frame('main'), childFrames: [] })).not.toThrow()
    expect(() => kit.assertNoChildFrames({ frame: frame('main'), childFrames: [{ frame: frame('kid', 'http://other.test/f') }] })).toThrow(`${REFUSAL}: found a frame http://other.test/f`)
    expect(() => kit.assertNoChildFrames({ frame: frame('main'), childFrames: [{ frame: frame('kid-7') }] })).toThrow(`${REFUSAL}: found a frame kid-7`)
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

  it('finds closed shadow roots anywhere in a pierced DOM tree and ignores the rest', () => {
    const node = (nodeName: string, extra: Record<string, unknown> = {}) => ({ nodeName, ...extra })
    const root = (shadowRootType: string) => node('#document-fragment', { shadowRootType, children: [node('H2')] })
    const page = (...kids: unknown[]) => node('#document', { children: [node('HTML', { children: kids })] })

    expect(kit.findClosedShadowRoot(page(node('DIV', { shadowRoots: [root('open'), root('user-agent')] })))).toBeNull()
    expect(kit.findClosedShadowRoot(page(node('TEMPLATE', { templateContent: page(node('P', { shadowRoots: [root('closed')] })) })))).toBeNull()
    expect(kit.findClosedShadowRoot(page(node('SECTION', { shadowRoots: [root('closed')] })))).toBe('SECTION')
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

// Byte-level fixtures: valid 64x64 GIF, PNG and APNG files (red, then blue) with one frame per playback delay (1/100 s), and minimal WebP/AVIF/JXL headers.
const SIZE = 64
const COLOURS = [[255, 0, 0], [0, 0, 255]]

const crc32 = (b: Buffer) => {
  let c = ~0

  for (const x of b) {
    c ^= x

    for (let k = 0; k < 8; k++) {c = (c >>> 1) ^ (0xedb88320 & -(c & 1))}
  }

  return ~c >>> 0
}

const be = (...n: number[]) => {
  const b = Buffer.alloc(4 * n.length)
  n.forEach((v, i) => b.writeUInt32BE(v, 4 * i))

  return b
}

const pngChunk = (type: string, data: Buffer) => {
  const body = Buffer.concat([Buffer.from(type, 'latin1'), data])

  return Buffer.concat([be(data.length), body, be(crc32(body))])
}

const pixels = (rgb: number[]) => deflateSync(Buffer.concat(Array.from({ length: SIZE }, () => Buffer.from([0, ...Array.from({ length: SIZE }, () => rgb).flat()]))))
const PNG_SIGNATURE = Buffer.from('\x89PNG\r\n\x1a\n', 'latin1')
const IHDR = pngChunk('IHDR', Buffer.concat([be(SIZE, SIZE), Buffer.from([8, 2, 0, 0, 0])]))

function apng(delays: number[], animated = true) {
  let seq = 0
  const parts = [PNG_SIGNATURE, IHDR, ...(animated ? [pngChunk('acTL', be(delays.length, 0))] : [])]

  delays.forEach((delay, i) => {
    if (animated) {parts.push(pngChunk('fcTL', Buffer.concat([be(seq++, SIZE, SIZE, 0, 0), Buffer.from([delay >> 8, delay & 255, 0, 100, 0, 0])])))}

    const data = pixels(COLOURS[i % 2])
    parts.push(i === 0 ? pngChunk('IDAT', data) : pngChunk('fdAT', Buffer.concat([be(seq++), data])))
  })

  return Buffer.concat([...parts, pngChunk('IEND', Buffer.alloc(0))])
}

const staticPng = () => apng([0], false)

function gif(delays: number[]) {
  const lzw = (index: number) => {
    const out: number[] = []
    let acc = 0, bits = 0

    for (const code of [...Array.from({ length: 2 * SIZE * SIZE }, (_, i) => (i % 2 ? index : 4)), 5]) {
      acc |= code << bits
      bits += 3

      while (bits >= 8) { out.push(acc & 255); acc >>= 8; bits -= 8 }
    }

    if (bits) {out.push(acc)}

    return Buffer.concat([...Array.from({ length: Math.ceil(out.length / 255) }, (_, i) => Buffer.from([Math.min(255, out.length - 255 * i), ...out.slice(255 * i, 255 * i + 255)])), Buffer.from([0])])
  }

  const frames = delays.map((delay, i) => Buffer.concat([Buffer.from([0x21, 0xf9, 4, 4, delay & 255, delay >> 8, 0, 0, 0x2c, 0, 0, 0, 0, SIZE, 0, SIZE, 0, 0, 2]), lzw(i % 2)]))

  return Buffer.concat([Buffer.from('GIF89a'), Buffer.from([SIZE, 0, SIZE, 0, 0x80, 0, 0, 255, 0, 0, 0, 0, 255]),
    Buffer.from('!\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00', 'latin1'), ...frames, Buffer.from(';')])
}

const le32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n);

 return b }

const webp = (flags: number) => {
  const chunk = (type: string, data: Buffer) => Buffer.concat([Buffer.from(type), le32(data.length), data])
  const body = Buffer.concat([Buffer.from('WEBP'), chunk('VP8X', Buffer.concat([Buffer.from([flags, 0, 0, 0]), Buffer.alloc(6)])), chunk('VP8L', Buffer.alloc(8))])

  return Buffer.concat([Buffer.from('RIFF'), le32(body.length), body])
}

const ftyp = (...brands: string[]) => Buffer.concat([be(8 + 8 * brands.length), Buffer.from('ftyp'), ...brands.map(b => Buffer.concat([Buffer.from(b), be(0)]))])

// The animation test of libwebp's own feature probe: a VP8X header with the animation flag, an ANIM chunk and one ANMF frame; and its lossless still counterpart.
const ANIMATED_WEBP = Buffer.from('UklGRlIAAABXRUJQVlA4WAoAAAASAAAAAAAAAAAAQU5JTQYAAAD/////AABBTk1GJgAAAAAAAAAAAAAAAAAAAGQAAABWUDhMDQAAAC8AAAAQBxAREYiI/gcA', 'base64')
const STATIC_WEBP = Buffer.from('UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==', 'base64')

const smil = (ms: number, id: string, extra = '') => `<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><rect id="${id}" y="20" width="20" height="20" fill="red"><animate attributeName="x" from="0" to="40" dur="${ms}ms" repeatCount="indefinite"/></rect>${extra}</svg>`
const stillSvg = (id: string) => `<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><title>${id}</title><rect width="64" height="64" fill="red"/></svg>`
const dataUri = (mime: string, body: Buffer | string) => `data:${mime};base64,${Buffer.from(body).toString('base64')}`
const SVG = 'image/svg+xml'
type ScrollKnobs = { x: number, y: number, onLoad: boolean }

// Prologs the old raw-text scan could not get past: a 70,000-character comment, and a DOCTYPE whose entity value holds "]>".
const PROLOGS: Record<string, string> = { 'long-comment': `<!-- ${'x'.repeat(70_000)} -->\n`, 'entity-doctype': '<!DOCTYPE svg [<!ENTITY closer "]>">]>\n' }

describe('animated resource detection', () => {
  it('counts GIF frames by walking the blocks, and reports 0 for anything that is not a GIF', () => {
    expect(kit.gifFrameCount(gif([10, 1000]))).toBe(2)
    expect(kit.gifFrameCount(gif([1, 1, 1, 1]))).toBe(4)
    expect(kit.gifFrameCount(gif([10]))).toBe(1)
    expect(kit.gifFrameCount(gif([10, 1000]).subarray(0, 60))).toBe(1)
    expect(kit.gifFrameCount(Buffer.from('GIF89a'))).toBe(0)
    expect(kit.gifFrameCount(staticPng())).toBe(0)
    expect(kit.gifFrameCount(Buffer.alloc(0))).toBe(0)
  })

  it('reads the acTL frame count of a PNG only when it precedes the image data, and 1 for an ordinary PNG', () => {
    expect(kit.apngFrameCount(apng([10, 1000]))).toBe(2)
    expect(kit.apngFrameCount(apng([5, 5, 5]))).toBe(3)
    expect(kit.apngFrameCount(apng([10]))).toBe(1)
    expect(kit.apngFrameCount(staticPng())).toBe(1)
    const idat = staticPng().subarray(PNG_SIGNATURE.length + IHDR.length)
    expect(kit.apngFrameCount(Buffer.concat([PNG_SIGNATURE, IHDR, idat.subarray(0, idat.length - 12), pngChunk('acTL', be(2, 0)), idat.subarray(idat.length - 12)]))).toBe(1)
    expect(kit.apngFrameCount(gif([10]))).toBe(0)
    expect(kit.apngFrameCount(PNG_SIGNATURE)).toBe(1)
  })

  it('reads the WebP animation flag from the VP8X header', () => {
    expect(kit.webpIsAnimated(webp(0x02))).toBe(true)
    expect(kit.webpIsAnimated(webp(0x12))).toBe(true)
    expect(kit.webpIsAnimated(webp(0x10))).toBe(false)
    expect(kit.webpIsAnimated(webp(0))).toBe(false)
    expect(kit.webpIsAnimated(ANIMATED_WEBP)).toBe(true)
    expect(kit.webpIsAnimated(STATIC_WEBP)).toBe(false)
    expect(kit.webpIsAnimated(gif([10, 10]))).toBe(false)
  })

  it('names the kind of an animated raster resource and nothing for still images; SVG is judged by the browser allowlist, not here', () => {
    expect(kit.animatedResource(gif([10, 1000]), 'image/gif')).toBe('gif')
    expect(kit.animatedResource(apng([10, 1000]), 'image/png')).toBe('apng')
    expect(kit.animatedResource(ANIMATED_WEBP, 'image/webp')).toBe('webp')
    expect(kit.animatedResource(gif([10, 1000]), 'application/octet-stream')).toBe('gif')
    expect(kit.animatedResource(Buffer.from(smil(1000, 'k')), SVG)).toBeNull()
    expect(kit.svgIsAnimated).toBeUndefined()

    for (const still of [[gif([10]), 'image/gif'], [staticPng(), 'image/png'], [apng([10]), 'image/png'], [STATIC_WEBP, 'image/webp'], [Buffer.from(stillSvg('s')), SVG],
      [Buffer.from([0xff, 0xd8, 0xff, 0xe0]), 'image/jpeg'], [Buffer.alloc(0), 'image/png'], [Buffer.from('<h1>hi</h1>'), 'text/html']] as const) {
      expect(kit.animatedResource(still[0], still[1]), still[1]).toBeNull()
    }
  })

  it('treats an AVIF image sequence as animated and a JPEG XL file as unverifiable, but a still AVIF as fine', () => {
    expect(kit.animatedResource(ftyp('avis', 'avis', 'mif1'), 'image/avif')).toBe('avif')
    expect(kit.animatedResource(ftyp('avif', 'mif1', 'avis'), 'image/avif')).toBe('avif')
    expect(kit.animatedResource(ftyp('avif', 'mif1', 'miaf'), 'image/avif')).toBeNull()
    expect(kit.animatedResource(Buffer.from([0xff, 0x0a, 1, 2, 3]), 'image/jxl')).toBe('jxl')
    expect(kit.animatedResource(Buffer.from([0, 0, 0, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a]), 'image/jxl')).toBe('jxl')
  })

  it('refuses with one sentence per kind of problem, naming every animated, unprovable and unreadable resource', () => {
    expect(() => kit.assertNoAnimatedResources([], [])).not.toThrow()
    expect(() => kit.assertNoAnimatedResources([], [], [])).not.toThrow()
    const long = `data:image/gif;base64,${'A'.repeat(100)}`
    const found = [{ kind: 'gif', url: 'http://127.0.0.1:1/a.gif', sha256: 'a'.repeat(64) }, { kind: 'apng', url: long, sha256: 'b'.repeat(64) }]
    expect(() => kit.assertNoAnimatedResources(found, [])).toThrow(`unsupported content: animated resources whose timelines cannot be frozen: gif http://127.0.0.1:1/a.gif (sha256 ${'a'.repeat(12)}), apng ${long.slice(0, 60)}… (sha256 ${'b'.repeat(12)})`)
    expect(() => kit.assertNoAnimatedResources([], [{ url: 'http://127.0.0.1:1/x.png' }])).toThrow('unsupported content: an image or document resource could not be read to check it for animation: http://127.0.0.1:1/x.png')

    const unprovable = [{ url: 'http://127.0.0.1:1/s.svg', sha256: 'c'.repeat(64) }, { url: long, sha256: 'd'.repeat(64) }]
    expect(() => kit.assertNoAnimatedResources([], [], unprovable)).toThrow(
      `unsupported content: the kit cannot prove that this SVG is static: http://127.0.0.1:1/s.svg (sha256 ${'c'.repeat(12)})\nunsupported content: the kit cannot prove that this SVG is static: ${long.slice(0, 60)}… (sha256 ${'d'.repeat(12)})`)
  })
})

describe('SVG data URLs a page references', () => {
  const MALFORMED = '<svg xmlns="http://www.w3.org/2000/svg"><rect></svg>'
  const b64 = (s: string) => `data:image/svg+xml;base64,${Buffer.from(s).toString('base64')}`
  const sha = (s: string | Buffer) => createHash('sha256').update(s).digest('hex')

  it('pulls the url("...") items out of computed style strings, unescaping quotes and backslashes, and ignores the rest', () => {
    expect(kit.cssUrls('none')).toEqual([])
    expect(kit.cssUrls('url("data:image/svg+xml;base64,AAA="), linear-gradient(red, blue), url("/x.png")')).toEqual(['data:image/svg+xml;base64,AAA=', '/x.png'])
    expect(kit.cssUrls('image-set(url("data:image/svg+xml,%3Csvg%2F%3E") 1x)')).toEqual(['data:image/svg+xml,%3Csvg%2F%3E'])
    expect(kit.cssUrls('url("a\\"b\\\\c")')).toEqual(['a"b\\c'])
    expect(kit.cssUrls(undefined)).toEqual([])
  })

  it('splits a srcset on its candidate separators without cutting a data URL at its own comma', () => {
    const d = 'data:image/svg+xml,%3Csvg%3E%3C/svg%3E'
    expect(kit.srcsetUrls(`${d} 1x, /b.png 2x`)).toEqual([d, '/b.png'])
    expect(kit.srcsetUrls(`  /a.png 100w ,\n ${d}   300w`)).toEqual(['/a.png', d])
    expect(kit.srcsetUrls(`/a.png, ${d}`)).toEqual(['/a.png', d])
    expect(kit.srcsetUrls('')).toEqual([])
  })

  it('decodes the exact bytes of an SVG data URL, base64 or percent-encoded, and returns null for anything else', () => {
    expect(kit.svgDataUrlBytes(b64(MALFORMED)).toString()).toBe(MALFORMED)
    expect(kit.svgDataUrlBytes(`data:image/svg+xml,${encodeURIComponent(MALFORMED)}`).toString()).toBe(MALFORMED)
    expect(kit.svgDataUrlBytes(`DATA:Image/SVG+XML;charset=utf-8;BASE64,${Buffer.from(MALFORMED).toString('base64')}`).toString()).toBe(MALFORMED)
    expect(kit.svgDataUrlBytes(`data:image/svg+xml;base64,${Buffer.from(MALFORMED).toString('base64').replace(/(.{8})/g, '$1 ')}`).toString()).toBe(MALFORMED)
    expect(kit.svgDataUrlBytes('data:image/svg+xml,%3Csvg%2F%3E#frag').toString()).toBe('<svg/>')
    expect([...kit.svgDataUrlBytes('data:image/svg+xml,%FF%00')]).toEqual([0xff, 0])

    for (const other of ['data:image/png;base64,AAAA', 'data:text/plain,<svg/>', 'data:image/svg+xml', '/a.svg', 'http://x/a.svg', 'blob:http://x/1', '']) {
      expect(kit.svgDataUrlBytes(other), other).toBeNull()
    }
  })

  const STYLED = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64"><style>rect{fill:rgb(0,255,0)}</style><rect width="64" height="64" fill="red"/></svg>'

  it('leaves what bytes hold to the browser: only a media type that may show them as SVG or XML (image/svg+xml, text/xml, application/xml, any +xml) makes unparseable bytes unprovable', () => {
    for (const mime of ['image/svg+xml', 'IMAGE/SVG+XML; charset=utf-8', 'text/xml', 'application/xml', 'text/xml; charset=UTF-8', 'application/xhtml+xml', 'application/vnd.example+xml']) {
      expect(kit.claimsXml(mime), mime).toBe(true)
    }

    for (const mime of ['application/octet-stream', 'text/plain', 'text/html', 'image/png', 'image/gif', 'font/woff2', 'application/json', 'text/xmlish', '', undefined]) {
      expect(kit.claimsXml(mime), String(mime)).toBe(false)
    }
  })

  it('decodes the exact bytes of a data URL labelled text/xml, application/xml or +xml, and of nothing else', () => {
    for (const type of ['text/xml', 'application/xml', 'application/xhtml+xml', 'TEXT/XML']) {
      expect(kit.svgDataUrlBytes(`data:${type};base64,${Buffer.from(STYLED).toString('base64')}`).toString(), type).toBe(STYLED)
      expect(kit.svgDataUrlBytes(`data:${type};charset=utf-8,${encodeURIComponent(STYLED)}`).toString(), type).toBe(STYLED)
    }

    for (const other of ['data:text/xml', `data:text/html,${encodeURIComponent(STYLED)}`, `data:text/plain,${encodeURIComponent(STYLED)}`, `data:image/png;base64,${Buffer.from(STYLED).toString('base64')}`]) {
      expect(kit.svgDataUrlBytes(other), other).toBeNull()
    }

    const rss = '<rss/>'
    const found = kit.svgDataResources({ urls: [`data:text/xml;base64,${Buffer.from(STYLED).toString('base64')}`, `data:application/xml,${encodeURIComponent(STYLED)}`, `data:text/xml,${encodeURIComponent(rss)}`] })
    expect(found.map((f: any) => f.url)).toEqual(['data:text/xml', 'data:text/xml'])
    expect(found.map((f: any) => f.sha256).sort()).toEqual([sha(STYLED), sha(rss)].sort())
  })

  it('hands every SVG data URL to the browser check whole, behind any prolog: a 70,000-character comment or a DOCTYPE entity holding "]>", under any SVG or XML media type', () => {
    for (const [name, prolog] of Object.entries(PROLOGS)) {
      const svg = prolog + STYLED

      for (const type of ['image/svg+xml', 'text/xml', 'application/xml']) {
        expect(sha(kit.svgDataUrlBytes(dataUri(type, svg)) ?? ''), `${name} ${type}`).toBe(sha(svg))
        expect(sha(kit.svgDataUrlBytes(`data:${type},${encodeURIComponent(svg)}`) ?? ''), `${name} ${type} percent`).toBe(sha(svg))
      }

      const found = kit.svgDataResources({ urls: [dataUri('text/xml', svg), dataUri('application/xml', svg)] })
      expect(found.map((f: any) => f.sha256), name).toEqual([sha(svg)])
    }
  })

  it('lists each distinct SVG data resource once, from every kind of reference, named by a short prefix and its sha256', () => {
    const other = '<svg xmlns="http://www.w3.org/2000/svg"><g></svg>'

    const found = kit.svgDataResources({
      urls: [b64(MALFORMED), '/plain.svg', b64(MALFORMED), 'data:image/png;base64,AAAA'],
      srcsets: [`/a.png 1x, ${b64(other)} 2x`],
      styles: [`url("data:image/svg+xml,${encodeURIComponent(MALFORMED)}"), none`, 'none'],
    })

    expect(found.map((f: any) => f.sha256).sort()).toEqual([sha(MALFORMED), sha(other)].sort())

    for (const f of found) {
      expect(f.url).toBe('data:image/svg+xml')
      expect(Buffer.isBuffer(f.bytes)).toBe(true)
    }

    expect(kit.svgDataResources({ urls: [], srcsets: [], styles: [] })).toEqual([])
  })
})

describe.skipIf(process.env.DESIGN_COMPARE_REAL_CHROMIUM !== '1')('real Chromium (opt-in, pinned binary from CHROME_PATH, shipped pin)', { concurrent: true }, () => {
  const chromium = process.env.CHROME_PATH as string
  let seq = 0

  // The tests run concurrently but capture one at a time: two browsers exhaust the 64 MB /dev/shm of a default container, and captures then stall or fail.
  let queue: Promise<unknown> = Promise.resolve()

  const limited = <T>(job: () => Promise<T>) => {
    const run = queue.then(job)
    queue = run.catch(() => {})

    return run
  }

  const capture = (options: Record<string, unknown>): Promise<any[]> => limited(() => kit.capture(options))

  const dirOf = (files: Record<string, string | Buffer>) => {
    const d = join(scratch, `case-${++seq}`)

    for (const [name, body] of Object.entries(files)) {
      mkdirSync(join(d, name, '..'), { recursive: true })
      writeFileSync(join(d, name), body)
    }

    return d
  }

  const memo = <T>(job: () => Promise<T>) => {
    let started: Promise<T> | undefined

    return () => (started ??= job())
  }

  // A browser launch costs far more than a page, so every launch captures many routes. A refusal ends a launch: the routes before it keep their files, and the
  // refusal (if any) is returned with them, so a launch holds any number of successful routes and at most one refused route, which comes last.
  const launchPages = (pages: Record<string, string>, extra: Record<string, string | Buffer> = {}) => {
    const outDir = join(scratch, `out-${++seq}`)
    const site = { ...Object.fromEntries(Object.entries(pages).map(([name, html]) => [`${name}.html`, html])), ...extra }

    return capture({ side: 'candidate', siteDir: dirOf(site), pages: Object.keys(pages).map(name => `/${name}`), outDir, chromium, runId: 'run-1', candidateDigest: 'd1' })
      .then(() => ({ outDir, error: undefined as any }), (error: any) => ({ outDir, error }))
  }

  const launchPrototype = (html: string) => {
    const file = join(scratch, `proto-${++seq}.html`)
    const outDir = join(scratch, `out-${++seq}`)
    writeFileSync(file, html)

    return capture({ side: 'prototype', prototypeFile: file, pages: ['/'], outDir, chromium, runId: 'r', candidateDigest: 'd' })
      .then(() => ({ outDir, error: undefined as any }), (error: any) => ({ outDir, error }))
  }

  const read = (outDir: string, width: number, route: string) => JSON.parse(readFileSync(join(outDir, `${kit.pageSlug(route)}-${width}.json`), 'utf8'))
  const pngOf = (outDir: string, width: number, route: string) => readFileSync(join(outDir, `${kit.pageSlug(route)}-${width}.png`))
  const byText = (bundle: any, text: string) => bundle.elements.find((e: any) => e.text === text)

  // Two routes of one launch hold the same evidence when everything but their own name is byte-identical.
  const sameEvidence = (run: { outDir: string }, a: string, b: string) => {
    for (const width of [1280, 375]) {
      const [x, y] = [a, b].map(route => JSON.stringify({ ...read(run.outDir, width, route), page: '', image: '' }))
      expect(x, `${a} ${b} ${width} bundle`).toBe(y)
      expect(pngOf(run.outDir, width, a).equals(pngOf(run.outDir, width, b)), `${a} ${b} ${width} image`).toBe(true)
    }
  }

  const receiver = async () => {
    const hits: string[] = []
    const server = createServer((req, res) => { hits.push(req.url as string); res.end('x') })
    server.on('upgrade', (req, socket) => { hits.push(`upgrade ${req.url}`); socket.destroy() })
    await new Promise<void>(r => server.listen(0, '127.0.0.1', r))

    return { base: `127.0.0.1:${(server.address() as { port: number }).port}`, hits, close: () => new Promise(r => server.close(r)) }
  }

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

  // ---- Motion and shadow roots: pairs of routes that differ only in a period, a speed or a rendering hint must give the same evidence. ----

  // One page per route: the animated tiny page, an animated open shadow root and a visible shadow root (with a canvas), then `native` (SMIL and marquee), the scroll fragment, which comes first so that its sticky box stays at 300 px, the rendering-normalisation fragment `nd`, and a shadow root on the body (first, before the rest of the page is parsed) whose animation period differs too.
  const motionPage = (ms: number, smoothing: string, native: string, scroll: ScrollKnobs, nd: string, bodyShadowMs: number) => `<!doctype html><meta charset=utf-8>
<style>body{margin:0;width:2400px;height:2400px;font:20px sans-serif;-webkit-font-smoothing:${smoothing}}
.sh{position:fixed;left:0;top:0;width:200px;height:50px;background:#abc}
.sa{position:sticky;top:0;margin-top:300px;width:200px;height:60px;background:#cba}
@keyframes spin{to{transform:rotate(360deg)}}
#box{width:80px;height:80px;background:#369;animation:spin ${ms}ms linear infinite}
#wa{width:80px;height:60px;background:red}h1{margin:16px}</style><body style="background:#eee">${bodyShadowScript(bodyShadowMs)}${scrollFragment(scroll)}<h1>Tiny page</h1><div id=box></div><div id=wa></div><p>Some text</p>
<div style="opacity:0"><p>Invisible text</p></div>
<div style="visibility:hidden"><p style="visibility:visible">Shown override</p></div><p>Visible control</p>
<header><h1>Site title</h1><nav><a href="/x" aria-label="Go home">Home</a></nav></header>
<main><section role="region"><h2>Part one</h2><p style="font:italic 700 18px/30px Georgia,serif;color:rgba(10,20,30,.5)">Body text</p>
<img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" alt="A dot" width=20 height=20><button>Go</button></section></main>
<div style="position:fixed;bottom:0;left:0;color:#369">Pinned</div>
<h1>Outside</h1><div id=animated></div><script>animated.attachShadow({mode:'open'}).innerHTML =
'<style>@keyframes m{to{transform:translateX(300px)}}h2{width:100px;animation:m ${ms}ms infinite linear}</style><h2>Inside</h2>'</script>
<div id=host></div><script>host.attachShadow({mode:'open'}).innerHTML = '<p style="font-size:10px">Visible shadow text</p><canvas width=50 height=20></canvas><p style="font-size:40px">Big shadow text</p>'</script>
${native}
${nd}
<script>document.getElementById('wa').animate([{transform:'translateX(0px)'},{transform:'translateX(300px)'}],{duration:${ms},iterations:Infinity})</script>`

  // Attaches an open shadow root to the body itself, with a slot so the page stays visible.
  const bodyShadowScript = (ms: number) => `<script>document.body.attachShadow({mode:'open'}).innerHTML =
'<style>@keyframes m{to{transform:translateX(300px)}}h2{width:100px;animation:m ${ms}ms infinite linear}</style><slot></slot><h2>Inside body</h2>'; document.currentScript.remove()</script>`

  const closedPage = '<div id=h></div><script>h.attachShadow({mode:"closed"}).innerHTML = "<h2>Hidden</h2>"</script>'

  // The routes before the refusal of a closed shadow root. The two routes differ in every period, speed and rendering hint, and must still give the same evidence.
  const motionLaunch = memo(() => {
    const [fast, slow] = [nativeMotion(1000, 40), nativeMotion(10_000, 1)]

    return launchPages({ 'motion-a': motionPage(1000, 'antialiased', fast.body, { x: 0, y: 0, onLoad: false }, ndFragment('subpixel-antialiased', 'optimizeSpeed'), 1000),
      'motion-b': motionPage(3700, 'subpixel-antialiased', slow.body, { x: 1000, y: 1000, onLoad: true }, ndFragment('none', 'optimizeLegibility'), 10_000), closed: closedPage },
      { 'acme.woff2': readFileSync(fileURLToPath(new URL('../web/public/fonts/Mondwest-Regular.woff2', import.meta.url))) })
  })

  it('gives identical bundles and images despite CSS animation, infinite Web Animations API motion and font smoothing', async () => {
    const run = await motionLaunch()
    sameEvidence(run, '/motion-a', '/motion-b')
    const bundle = read(run.outDir, 1280, '/motion-a')
    expect(contractProblems(bundle)).toEqual([])
    expect(bundle.elements.find((e: any) => e.box.width === 80 && e.box.height === 60).box.x).toBe(0)
    expect(bundle.chromium_version).toBe('153.0.8010.0')
  }, 120_000)

  it('emits bundles that satisfy the C1 contract at both widths: landmarks, role, label, alt, parent, fixed and the STYLE_FIELDS names', async () => {
    const run = await motionLaunch()

    for (const width of [1280, 375]) {
      const bundle = read(run.outDir, width, '/motion-a')
      expect(bundle.width).toBe(width)
      expect(contractProblems(JSON.parse(JSON.stringify(bundle))), `${width}`).toEqual([])
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
    const keys = judgeKeys(read((await motionLaunch()).outDir, 1280, '/motion-a').elements)
    expect(keys).toContain('/p:"visible control"#1')
    expect(keys).toContain('/p:"shown override"#1')
    expect(keys.join('\n')).not.toContain('invisible text')
  }, 120_000)

  it('gives identical bundles and pixels for a shadow-root element with an infinite CSS animation when only its period differs', async () => {
    const run = await motionLaunch()
    sameEvidence(run, '/motion-a', '/motion-b')
    expect(byText(read(run.outDir, 1280, '/motion-a'), 'inside').tag).toBe('h2')
  }, 120_000)

  it('measures visible shadow-root text with its own type values and page coordinates', async () => {
    const run = await motionLaunch()
    const bundle = read(run.outDir, 1280, '/motion-a')

    expect(byText(bundle, 'visible shadow text').style['font-size']).toBe(10)
    expect(byText(bundle, 'big shadow text').style['font-size']).toBe(40)
    expect(byText(bundle, 'big shadow text').box.height).toBeGreaterThan(byText(bundle, 'visible shadow text').box.height)
    expect(bundle.masks).toContainEqual(expect.objectContaining({ width: 50, height: 20 }))
  }, 120_000)

  it('refuses a closed shadow root instead of claiming complete evidence', async () => {
    expect((await motionLaunch()).error?.message).toMatch(/closed shadow root/)
  }, 120_000)

  // ---- Rendering normalisation: what the page can read, set and retain. ----

  const normFragment = (smoothing: string, rendering: string, more = '') => {
    const decl = `-webkit-font-smoothing:${smoothing};text-rendering:${rendering}`
    const reflect = (id: string, out: string) => `${out}.textContent = getComputedStyle(${id}).webkitFontSmoothing + ' / ' + getComputedStyle(${id}).textRendering`

    return `Top body text<p id=na style="${decl}!important">Inline</p><p id=nra></p><script>${reflect('na', 'nra')}</script>
<div id=rhost></div><script>const r = rhost.attachShadow({mode:'open'}), own = new CSSStyleSheet(); own.replaceSync('h2{${decl}}')
r.adoptedStyleSheets = [own]; r.innerHTML = '<h2 id=s>S</h2><p id=o></p>'
const o = r.getElementById('o'); o.textContent = getComputedStyle(r.getElementById('s')).webkitFontSmoothing + ' / ' + getComputedStyle(r.getElementById('s')).textRendering</script>
<div id=dh><template shadowrootmode=open><style>h3{${decl}}</style><h3 id=d>D</h3><p id=e></p></template></div>
<script>const dr = dh.shadowRoot; dr.getElementById('e').textContent = getComputedStyle(dr.getElementById('d')).webkitFontSmoothing + ' / ' + getComputedStyle(dr.getElementById('d')).textRendering</script>
${retainFragment(false)}${more}`
  }

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

  const dynamicFragment = (smoothing: string, rendering: string) => {
    const make = dynamic(smoothing, rendering)

    return `<div id=dtop></div><div id=dhost></div><script>
${make}(document.getElementById('dtop'))
dhost.attachShadow({mode:'open'}); ${make}(dhost.shadowRoot)</script>`
  }

  // Three texts at the same place, set in the same face: its name quoted with a double and with a single quote, a name that is not loaded, and the loaded name in other case.
  const fontFragment = `<style>
@font-face{font-family:"Acme,Display";src:url(/acme.woff2) format('woff2')}
@font-face{font-family:'Acme,Display';src:url(/acme.woff2) format('woff2')}
.fonts{position:relative;font-size:32px}.fonts p{position:absolute;left:0;top:0;margin:0}</style>
<div class=fonts><p style='font-family:"Acme,Display", serif'>Measured text</p><p style='font-family:"Acme,Other", serif'>Measured text</p><p style="font-family:'ACME,display', monospace">Measured text</p></div>`

  // The fragment of the motion pages that holds the rendering-hint reads, the synchronous-normalisation paths and the font faces.
  const ndFragment = (smoothing: string, rendering: string) => normFragment(smoothing, rendering, `${dynamicFragment(smoothing, rendering)}${fontFragment}`)

  // A mutation path the kit cannot cover. With a retained computed style it must be refused; without one it is accepted and the page reads the normalised value.
  const retainFragment = (retain: boolean) => `<p id=ka>A</p><p id=ko></p><script>(() => {
const p = document.getElementById('ka'), live = ${retain ? 'getComputedStyle(p)' : 'null'}
p.attributes.style ? 0 : p.setAttribute('style', 'color:red')
p.attributes.style.value = 'text-rendering:optimizeSpeed!important'
document.getElementById('ko').textContent = (live || getComputedStyle(p)).textRendering })()</script>`

  const retainPage = (retain: boolean) => `<!doctype html>${retainFragment(retain)}`

  // Every paragraph that reads back the normalised values shows them as this text: 2 from the enforcement page, 3 from the first group of reads, 12 from the synchronous paths.
  const NORMALISED = 'antialiased / geometricprecision'

  const normaliseLaunch = memo(() => launchPages({ 'retain-on': retainPage(true) }))

  it('visits a body-attached open shadow root with an animated h2 and gives period-independent images', async () => {
    const run = await motionLaunch()
    sameEvidence(run, '/motion-a', '/motion-b')

    for (const width of [1280, 375]) {expect(byText(read(run.outDir, width, '/motion-a'), 'inside body').tag).toBe('h2')}
  }, 120_000)

  it('normalises before page code reads it and measures the body: one synchronous computed-style read of inline !important, a shadow root and a declarative shadow root', async () => {
    const run = await motionLaunch()
    sameEvidence(run, '/motion-a', '/motion-b')

    for (const width of [1280, 375]) {
      const bundle = read(run.outDir, width, '/motion-a')
      expect(bundle.elements.filter((e: any) => e.text === NORMALISED), `${width}`).toHaveLength(2 + 3 + 12)
      const bodies = bundle.elements.filter((e: any) => e.tag === 'body')
      expect(bodies.map((e: any) => e.text), `${width} body text`).toEqual(['top body text'])
      expect(bodies.find((e: any) => e.text === 'top body text').style['background-color']).toEqual([238, 238, 238, 1])
    }
  }, 120_000)

  it('normalises synchronously: created, style-set (setProperty, cssText, style attribute, style property), inserted and read-back elements in the page and an open shadow root', async () => {
    const run = await motionLaunch()
    sameEvidence(run, '/motion-a', '/motion-b')

    for (const width of [1280, 375]) {
      expect(read(run.outDir, width, '/motion-a').elements.filter((e: any) => e.text === NORMALISED), `${width}`).toHaveLength(2 + 3 + 12)
    }
  }, 120_000)

  it('refuses the capture when a retained computed style was read after a mutation path the kit cannot cover, and accepts the same bypass when nothing observed the element', async () => {
    const [run, motion] = [await normaliseLaunch(), await motionLaunch()]
    expect(run.error?.message).toContain('normalisation could not be guaranteed')

    for (const width of [1280, 375]) {expect(read(motion.outDir, width, '/motion-a').elements.some((e: any) => e.text === 'geometricprecision'), `${width}`).toBe(true)}
  }, 120_000)

  const forced = '-webkit-font-smoothing:subpixel-antialiased!important;text-rendering:optimizeSpeed!important'

  const enforcePage = (script: string) => `<style>#a{${forced}}</style><h1 id=a>AVAVATAR</h1><p id=b style="${forced}">Inline</p><p id=ra></p><p id=rb></p>
<script>const show = () => { for (const [s, d] of [[a, ra], [b, rb]]) { const c = getComputedStyle(s); d.textContent = c.webkitFontSmoothing + ' / ' + c.textRendering } ${script} requestAnimationFrame(show) }; show()</script>`

  // ---- Scroll, native motion and tiling, ending in the refusal of the page that empties the normalisation layer. ----

  // Fixed and sticky boxes; the page is scrolled while loading. Its styles are the .sh and .sa rules of the motion page.
  const scrollFragment = ({ x, y, onLoad }: ScrollKnobs) => {
    const scroll = `scrollTo(${x},${y})`

    return `<header class=sh>Fixed</header><aside class=sa>Sticky</aside><main>Control</main>
<script>${onLoad ? `addEventListener('load', () => ${scroll})` : scroll}</script>${enforcePage(`b.style.setProperty('-webkit-font-smoothing', 'subpixel-antialiased', 'important');`)}`
  }

  const enforceLaunch = memo(() => launchPages({
    'enforce-empty': `${enforcePage('')}<script>addEventListener('load', () => document.getElementById('__dc_style').textContent = '')</script>`,
  }))

  it('enforces the rendering values against ID-selector declarations and inline !important set again by the page, and fails when the page empties the layer', async () => {
    const [run, motion] = [await enforceLaunch(), await motionLaunch()]

    for (const width of [1280, 375]) {
      expect(read(motion.outDir, width, '/motion-b').elements.filter((e: any) => e.text === NORMALISED), `${width}`).toHaveLength(2 + 3 + 12)
    }

    expect(run.error?.message).toContain('rendering normalisation is not effective')
  }, 120_000)

  it('measures fixed and sticky boxes in viewport coordinates, the rest in document coordinates, and captures the same page whatever the initial scroll', async () => {
    const run = await motionLaunch()
    sameEvidence(run, '/motion-a', '/motion-b')

    for (const width of [1280, 375]) {
      const bundle = read(run.outDir, width, '/motion-b')
      expect(byText(bundle, 'fixed')).toMatchObject({ fixed: true, box: { x: 0, y: 0, width: 200, height: 50 } })
      expect(byText(bundle, 'sticky')).toMatchObject({ fixed: true, box: { x: 0, y: 300, width: 200, height: 60 } })
      expect(byText(bundle, 'control').fixed).toBe(false)
    }
  }, 120_000)

  const nativeMotion = (ms: number, amount: number) => {
    const root = (kids: string) => `<svg xmlns="http://www.w3.org/2000/svg" width="360" height="120" style="display:block">${kids}</svg>`

    const shapes = `<rect y="20" width="60" height="40" fill="red"><animate attributeName="x" from="0" to="280" dur="${ms}ms" repeatCount="indefinite"/></rect>
<rect x="100" y="70" width="40" height="40" fill="green"><animateTransform attributeName="transform" type="rotate" from="0 120 90" to="360 120 90" dur="${ms}ms" repeatCount="indefinite"/></rect>
<circle r="10" cy="100" fill="blue"><animateMotion path="M200,0 L340,0" dur="${ms}ms" repeatCount="indefinite"/></circle>
<rect x="300" y="20" width="30" height="30" fill="black"><set attributeName="fill" to="white" begin="${ms / 2}ms"/></rect>`

    const marquees = `<marquee scrollamount="${amount}" style="display:block;width:300px">Moving text control</marquee><marquee direction="up" height="60" scrollamount="${amount}">Up</marquee>`

    return { body: `<h1>Native motion</h1>${root(shapes)}${marquees}<div id=nhost></div>
<script>document.getElementById('nhost').attachShadow({mode:'open'}).innerHTML = '${root(shapes)}<marquee scrollamount="${amount}">Shadow mover</marquee>'</script>` }
  }

  const tilesPage = (height: number, label: string) => `<!doctype html><style>html,body{margin:0}
#s{box-sizing:border-box;padding-top:300px;height:${height}px;background:repeating-linear-gradient(#cc3333 0 50px,#3399cc 50px 100px)}
#k{position:sticky;top:0;width:200px;height:40px;background:#00aa00}
#h{position:fixed;top:0;left:0;width:80px;height:60px;background:#ff0000}</style><div id=s><div id=k></div></div><div id=h></div><h1 style="position:absolute;left:100px;top:100px;margin:0;font:20px sans-serif">${label}</h1>`

  const declarativePage = '<div><template shadowrootmode="closed"><h2>Hidden</h2></template></div>'

  const nativeLaunch = memo(() => launchPages({ declarative: declarativePage }))

  const cliSite = dirOf({ 'a-b.html': tilesPage(836, 'Dash route'), 'a/b.html': tilesPage(1750, 'Slash route') })
  const cliOut = join(scratch, 'cli-e2e')

  // A run that is refused for overwriting stops before it starts a browser, so only a run that captures waits for its turn in the queue.
  const cli = (pages: string, outDir = cliOut, captures = true) => (captures ? limited : <T>(job: () => Promise<T>) => job())(() => new Promise<{ status: number | null, stderr: string }>(done => {
    const child = spawn(process.execPath, [captureFile, '--side', 'candidate', '--site', cliSite, '--pages', pages,
      '--out', outDir, '--run-id', 'r', '--candidate-digest', 'd'], { env: { ...process.env, CHROME_PATH: chromium } })

    let stderr = ''
    child.stderr.on('data', d => { stderr += d })
    child.on('close', status => done({ status, stderr }))
  }))

  const cliRun = memo(() => cli('/a-b /a/b'))

  it('freezes SMIL animate, animateTransform, animateMotion and set, and stops marquee, in the page and an open shadow root, whatever their duration or speed', async () => {
    const run = await motionLaunch()
    sameEvidence(run, '/motion-a', '/motion-b')
    const png = decodePng(pngOf(run.outDir, 1280, '/motion-a'))
    const top = read(run.outDir, 1280, '/motion-a').elements.find((e: any) => e.tag === 'svg').box.y
    expect(png.at(30, Math.round(top) + 40), 'animate rect at its start').toEqual([255, 0, 0])
  }, 120_000)

  it('keeps first-tile pixels when the last tile overlaps and fixed or sticky elements are hidden', async () => {
    const run = await cliRun()
    expect(run.status, run.stderr).toBe(0)
    const stripe = (y: number) => (Math.floor(y / 50) % 2 ? [51, 153, 204] : [204, 51, 51])

    for (const [route, height] of [['/a-b', 836], ['/a/b', 1750]] as const) {
      for (const width of [1280, 375]) {
        const png = decodePng(pngOf(cliOut, width, route))
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

  it('refuses declarative closed shadow roots instead of claiming complete evidence', async () => {
    expect((await nativeLaunch()).error?.message).toMatch(/closed shadow root/)
  }, 120_000)

  // ---- Fonts, then the refusal of embedded documents and a page that holds the scroll away from the top. ----

  it('keeps comma-containing quoted font families apart on the measured element, and equal when the names match', async () => {
    const run = await motionLaunch()

    for (const width of [1280, 375]) {
      const [a, b, c, ...more] = read(run.outDir, width, '/motion-a').elements.filter((e: any) => e.text === 'measured text')
      expect(more, `${width}`).toEqual([])
      const first = (e: any) => probe.firstFamily(e.style['font-family'])
      expect(a.style['font-family'], `${width}`).toBe('"Acme,Display", serif')
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

  // Each refused page ends its launch, so every kind of embedded document has its own. The pages are refused with one plain message that names the element or the frame.
  const embeddedLaunches = {
    iframe: memo(() => launchPages({ iframe: '<h1>Single</h1><iframe srcdoc="<h2>Inner</h2>"></iframe>' })),
    object: memo(() => launchPages({ object: '<h1>Single</h1><object data="data:text/html,%3Ch2%3EInner%3C/h2%3E" type="text/html"></object>' })),
    embed: memo(() => launchPages({ embed: '<h1>Single</h1><embed src="data:text/html,%3Ch2%3EInner%3C/h2%3E" type="text/html">' })),
    sandboxed: memo(() => launchPages({ sandboxed: '<h1>Single</h1><iframe sandbox srcdoc="<h2>Opaque</h2>"></iframe>' })),
    crossSite: memo(() => launchPages({ cross: '<h1>Single</h1><iframe src="http://localhost:9/other"></iframe>' })),
    prototype: memo(() => launchPrototype('<h1>Proto</h1><iframe srcdoc="<h2>Inner</h2>"></iframe>')),
  }

  // The page embeds a frame only once the kit has scrolled it to the second tile, after the checks at the start and before the first screenshot have passed.
  const lateLaunch = memo(() => launchPages({
    late: `<!doctype html><style>body{margin:0;height:2000px}</style><h1>Late</h1><script>
addEventListener('scroll', () => { if (scrollY > 0 && !document.querySelector('iframe')) document.body.appendChild(document.createElement('iframe')) })</script>`,
  }))

  it.each(['iframe', 'object', 'embed'] as const)('refuses a candidate page with an <%s> element and names it', async name => {
    const run = await embeddedLaunches[name]()
    expect(run.error?.message).toBe(`${REFUSAL}: found <${name}>`)
    expect(readdirSync(run.outDir)).toEqual([])
  }, 120_000)

  it('refuses a sandboxed or cross-site frame whose document the kit cannot read, by the same sentence', async () => {
    expect((await embeddedLaunches.sandboxed()).error?.message).toMatch(new RegExp(`^${REFUSAL}: found <iframe>`))
    expect((await embeddedLaunches.crossSite()).error?.message).toMatch(new RegExp(`^${REFUSAL}: found (<iframe>|a frame )`))
  }, 120_000)

  it('refuses a prototype with an embedded document, by the same sentence', async () => {
    const run = await embeddedLaunches.prototype()
    expect(run.error?.message).toBe(`${REFUSAL}: found <iframe>`)
    expect(readdirSync(run.outDir)).toEqual([])
  }, 120_000)

  it('refuses a page that gains a frame after the first checks, before it is screenshot', async () => {
    const run = await lateLaunch()
    expect(run.error?.message).toMatch(new RegExp(`^${REFUSAL}: found (<iframe>|a frame )`))
    expect(readdirSync(run.outDir)).toEqual([])
  }, 120_000)

  const stuckLaunch = memo(() => launchPages({
    stuck: `<!doctype html><style>body{margin:0;height:3000px}</style><h1>Stuck</h1><script>
addEventListener('scroll', () => { if (scrollY === 0) scrollTo(0, 400) }); scrollTo(0, 400)</script>`,
  }))

  it('refuses the capture when the page keeps the scroll away from the top-of-page state', async () => {
    expect((await stuckLaunch()).error?.message).toContain('top-of-page scroll state')
  }, 120_000)

  // ---- Requests that leave the page: one launch on each side, and receivers that must never be reached. ----

  const outside = memo(async () => {
    const r = await receiver()
    const s = await stunTurn()
    const ice = `{iceServers:[{urls:'${s.stun}'},{urls:'${s.turn}',username:'u',credential:'c'}]}`

    const network = `<h1>net</h1><script>
for (const f of [() => new WebSocket('ws://${r.base}/socket'), () => new EventSource('http://${r.base}/es'),
  () => navigator.sendBeacon('http://${r.base}/beacon', 'x'), () => new Worker('http://${r.base}/w.js'), () => new Worker('/w.js'),
  () => new SharedWorker('/w.js'), () => navigator.serviceWorker.register('/w.js'),
  () => fetch('http://${r.base}/fetch').catch(() => {})]) { try { f() } catch {} }</script>`

    const popup = `<h1>popup</h1><script>window.open('http://${r.base}/popup')</script>`
    const realm = `<h1>realm</h1><script>const p = window.open('about:blank'); new p.WebSocket('ws://${r.base}/clean-socket')</script>`

    const run = `for (const C of [RTCPeerConnection, webkitRTCPeerConnection]) { try { const pc = new C(${ice}); pc.createDataChannel('x'); pc.createOffer().then(o => pc.setLocalDescription(o)).catch(() => {}) } catch {} }
try { new WebTransport('https://127.0.0.1:9/') } catch {}`

    const rtc = `<h1>rtc</h1><script>${run}</script>`

    const routes = `<h1>routes</h1><a id=a target=_blank href="http://${r.base}/link">l</a><form id=f target=_blank action="http://${r.base}/form"></form><script>
for (const f of [() => a.click(), () => f.submit()]) { try { f() } catch {} }</script>`

    const leaks = `<link rel=stylesheet href="asset.css"><link rel=stylesheet href="/missing-required.css"><link rel=stylesheet href="http://${r.base}/x.css">`

    const [candidate, prototype] = await Promise.all([
      launchPages({ net: `<!doctype html>${network}${popup}${realm}${rtc}`, routes: `<!doctype html>${routes}` },
        { 'w.js': `new WebSocket('ws://${r.base}/from-worker')` }),
      launchPrototype(`<!doctype html>${network}${popup}${realm}${rtc}${leaks}`),
    ])

    await new Promise(done => setTimeout(done, 1000))
    await Promise.all([r.close(), s.close()])

    return { candidate, prototype, r, s }
  })

  it('refuses outside WebSocket, EventSource, beacon, workers and fetch before they reach a separate receiver', async () => {
    const { candidate, prototype, r } = await outside()
    const wanted = [`ws://${r.base}/socket`, `http://${r.base}/es`, `http://${r.base}/beacon`, `http://${r.base}/w.js`, `http://${r.base}/fetch`]

    for (const width of [1280, 375]) {
      const { blocked_requests: blocked } = read(candidate.outDir, width, '/net')
      expect(blocked).toEqual(expect.arrayContaining(wanted))
      expect(blocked.some((u: string) => u.endsWith('/w.js') && !u.includes(r.base))).toBe(true)
    }

    expect(prototype.error.message).toBe('the chosen design file is not self-contained')
    expect(prototype.error.blocked).toEqual(expect.arrayContaining(wanted))
    expect(r.hits).toEqual([])
  }, 120_000)

  it('stops a prototype that references same-origin or outside resources', async () => {
    const { prototype, r } = await outside()
    expect(prototype.error.message).toBe('the chosen design file is not self-contained')

    for (const suffix of ['/__prototype__/asset.css', '/missing-required.css', `${r.base}/x.css`]) {
      expect(prototype.error.blocked.some((u: string) => u.endsWith(suffix)), suffix).toBe(true)
    }

    expect(r.hits).toEqual([])
  }, 120_000)

  it('never lets direct popup navigation or an about:blank popup native WebSocket reach a separate receiver, on the candidate and on the prototype side', async () => {
    const { candidate, prototype, r } = await outside()

    for (const width of [1280, 375]) {
      const { blocked_requests: blocked } = read(candidate.outDir, width, '/net')
      expect(blocked, `popup ${width}`).toContain(`http://${r.base}/popup`)
      expect(blocked, `realm ${width}`).toContain('about:blank')
    }

    expect(prototype.error.blocked).toEqual(expect.arrayContaining([`http://${r.base}/popup`, 'about:blank']))
    expect(r.hits).toEqual([])
  }, 120_000)

  it('also keeps link and form routes from reaching a separate receiver', async () => {
    const { r } = await outside()
    expect(r.hits).toEqual([])
  }, 120_000)

  it('refuses WebRTC, WebTransport and data channels in the page before ICE reaches a STUN or TURN receiver, on both sides', async () => {
    const { candidate, prototype, s } = await outside()

    for (const width of [1280, 375]) {
      expect(read(candidate.outDir, width, '/net').blocked_requests).toEqual(expect.arrayContaining([s.stun, s.turn, 'webrtc:RTCPeerConnection', 'https://127.0.0.1:9/']))
    }

    expect(prototype.error.blocked).toEqual(expect.arrayContaining([s.stun, s.turn]))
    expect(s.seen).toEqual({ packets: 0, connections: 0 })
  }, 120_000)

  const protoOk = memo(() => launchPrototype('<style>h1{color:#369}</style><h1>P</h1><img alt=dot src="data:image/gif;base64,R0lGODlhAQABAAAAACw="><svg width=10 height=10><rect width=10 height=10 /></svg><canvas width=10 height=10></canvas><video width=10 height=10></video>'))

  it('accepts an inline prototype, also with an img, svg, canvas and video', async () => {
    const run = await protoOk()
    expect(run.error).toBeUndefined()
    expect(read(run.outDir, 1280, '/').masks).toHaveLength(2)

    for (const width of [1280, 375]) {expect(read(run.outDir, width, '/').blocked_requests).toEqual([])}
  }, 120_000)

  // ---- The shipped CLI. ----

  it('writes one bundle and PNG per route and width through the shipped CLI, never overwriting', async () => {
    const run = await cliRun()
    expect(run.status, run.stderr).toBe(0)
    const files = readdirSync(cliOut)
    expect(files.filter(f => f.endsWith('.json'))).toHaveLength(4)
    expect(files.filter(f => f.endsWith('.png'))).toHaveLength(4)

    for (const [route, heading] of [['/a-b', 'dash route'], ['/a/b', 'slash route']]) {
      for (const width of [1280, 375]) {
        const bundle = read(cliOut, width, route)
        expect(bundle.page).toBe(route)
        expect(bundle.image).toBe(`${kit.pageSlug(route)}-${width}.png`)
        expect(bundle.elements.map((e: any) => e.text)).toContain(heading)
        expect(files).toContain(bundle.image)
      }
    }

    const again = await cli('/a-b', cliOut, false)
    expect(again.status).toBe(1)
    expect(again.stderr).toContain('refusing to overwrite')
    expect(readdirSync(cliOut)).toHaveLength(8)
    const dup = await cli('/a-b /a-b', join(scratch, 'cli-dup'), false)
    expect(dup.status).toBe(1)
    expect(dup.stderr).toContain('refusing to overwrite')
  }, 120_000)

  // ---- SVG resources: the allowlist. ----

  const stillRaster = { 'still.gif': gif([10]), 'still.png': staticPng(), 'single.png': apng([10]), 'still.webp': STATIC_WEBP }
  const SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'
  const svgDoc = (inner: string) => `<svg ${SVG_NS} width="64" height="64">${inner}</svg>`
  const fill = '<rect width="64" height="64" fill="red"/>'
  const GRADIENT = `<?xml version="1.0" encoding="UTF-8"?>${svgDoc(`<title>gradient</title><desc>path and gradient</desc><defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#f00"/><stop offset="1" stop-color="#00f"/></linearGradient></defs><path d="M0 0H64V64H0Z" fill="url(#g)"/>`)}`

  // Each of these is outside the allowlist for its own reason; the last-but-five writes @keyframes with a character reference.
  const unsupportedSvgs: Record<string, string> = {
    'style-element.svg': svgDoc(`<style>rect{fill:red}</style>${fill}`),
    'style-attribute.svg': svgDoc('<rect width="64" height="64" style="fill:red"/>'),
    'animate-element.svg': smil(1000, 'a'),
    'processing-instruction.svg': `<?xml-stylesheet type="text/css" href="data:text/css,rect%7Bfill:red%7D"?>${svgDoc(fill)}`,
    'reference-keyframes.svg': svgDoc(`<style>&#64;keyframes k{to{x:1}}rect{animation:k 1s infinite}</style>${fill}`),
    'event-attribute.svg': svgDoc('<rect width="64" height="64" onclick="x()"/>'),
    'unlisted-element.svg': svgDoc('<use href="#u"/>'),
    'background-style.svg': svgDoc(`<style>rect{fill:red}</style><title>background</title>${fill}`),
    'mask-style.svg': svgDoc(`<style>rect{fill:red}</style><title>mask</title>${fill}`),
    'border-style.svg': svgDoc(`<style>rect{fill:red}</style><title>border</title>${fill}`),
    'long-comment.svg': PROLOGS['long-comment'] + svgDoc(`<style>rect{fill:red}</style><title>long comment</title>${fill}`),
    'entity-doctype.svg': PROLOGS['entity-doctype'] + svgDoc(`<style>rect{fill:red}</style><title>entity doctype</title>${fill}`),
  }

  const STATIC_PAGE = `<!doctype html><style>body{margin:0}div{width:64px;height:64px;background:url(${dataUri(SVG, stillSvg('red'))})}img{display:block}</style>
<img alt="" src="${dataUri(SVG, stillSvg('red'))}"><div></div><img alt="" src="/gradient.svg"><img alt="" src="${dataUri(SVG, GRADIENT)}"><img alt="" src="data:image/svg+xml,${encodeURIComponent(GRADIENT)}"><img alt="" src="/still.gif"><img alt="" src="/still.png"><img alt="" src="/single.png"><img alt="" src="/still.webp"><h1>Static</h1>`

  // Every other browser consumer of a data: URL that can hold an SVG document or image, each with its own malformed bytes.
  const malformedFor = (name: string) => `<svg ${SVG_NS}><rect id="${name}"></svg>`
  const consumerUri = (name: string) => dataUri(SVG, malformedFor(name))

  const otherConsumers: Array<[string, string]> = [
    ['poster', `<video poster="${consumerUri('poster')}"></video>`],
    ['use', `<svg width="8" height="8"><use href="${consumerUri('use')}#u"/></svg>`],
    ['cursor', `<div style="cursor:url(${consumerUri('cursor')}) 0 0,auto"></div>`],
    ['shape', `<div style="float:left;shape-outside:url(${consumerUri('shape')})"></div>`],
    ['filter', `<div style="filter:url(${consumerUri('filter')}#f)"></div>`],
    ['clip', `<div style="clip-path:url(${consumerUri('clip')}#c)"></div>`],
    ['maskbox', `<div style="-webkit-mask-box-image-source:url(${consumerUri('maskbox')})"></div>`],
    ['fill', `<svg width="8" height="8"><rect width="8" height="8" style="fill:url(${consumerUri('fill')}#p)"/></svg>`],
  ]

  const REFUSED_PAGE = `<!doctype html><style>body{margin:0}img,div{display:block;width:64px;height:64px}
#bg{background:url(/background-style.svg)}#mask{background:red;mask-image:url(/mask-style.svg)}#border{border-image:url(/border-style.svg) 1}</style><h1>Refused</h1>
${['style-element', 'style-attribute', 'animate-element', 'processing-instruction', 'reference-keyframes', 'event-attribute', 'unlisted-element', 'long-comment', 'entity-doctype'].map(n => `<img alt="" src="/${n}.svg">`).join('')}
<div id=bg></div><div id=mask></div><div id=border></div>
<img alt="" src="/gradient.svg"><img alt="" src="/a.gif"><img alt="" src="/a.png"><img alt="" src="/a.webp"><img alt="" src="/still.gif"><img alt="" src="/still.webp">
${otherConsumers.map(([, html]) => html).join('')}`

  // One browser launch: two identical static routes, then the route that is refused (a refusal ends the launch, so it comes last).
  const svgLaunch = memo(() => launchPages({ 'static-1': STATIC_PAGE, 'static-2': STATIC_PAGE, refused: REFUSED_PAGE }, {
    'gradient.svg': GRADIENT, ...unsupportedSvgs, 'a.gif': gif([10, 1000]), 'a.png': apng([10, 1000]), 'a.webp': ANIMATED_WEBP, ...stillRaster,
  }))

  it('captures a static SVG (path and gradient) as img and background, with single-frame rasters, identically on two runs, keeping the pixels', async () => {
    const run = await svgLaunch()
    sameEvidence(run, '/static-1', '/static-2')
    expect(decodePng(pngOf(run.outDir, 1280, '/static-1')).at(30, 30)).toEqual([255, 0, 0])
    expect(contractProblems(read(run.outDir, 1280, '/static-1'))).toEqual([])
  }, 120_000)

  it('refuses an SVG img with a style element, a style attribute, an animate element, a processing instruction or a style element writing @keyframes by character reference', async () => {
    const message: string = (await svgLaunch()).error?.message ?? ''
    const lines = message.split('\n').filter(l => l.startsWith('unsupported content: the kit cannot prove that this SVG is static'))
    expect(lines.length, message).toBeGreaterThan(0)

    for (const name of ['style-element', 'style-attribute', 'animate-element', 'processing-instruction', 'reference-keyframes', 'event-attribute', 'unlisted-element', 'long-comment', 'entity-doctype']) {
      expect(lines.some(l => l.includes(`/${name}.svg`)), `${name}: ${message}`).toBe(true)
    }

    expect(message).not.toContain('gradient.svg')
  }, 120_000)

  it('refuses a style-element SVG as a CSS background, a mask and a border image', async () => {
    const message: string = (await svgLaunch()).error?.message ?? ''

    for (const name of ['background-style', 'mask-style', 'border-style']) {
      expect(message, name).toMatch(new RegExp(`cannot prove that this SVG is static: [^\\n]*/${name}\\.svg \\(sha256 [0-9a-f]{12}\\)`))
    }
  }, 120_000)

  it('still refuses animated GIF, APNG and WebP resources and lets single-frame ones through', async () => {
    const message: string = (await svgLaunch()).error?.message ?? ''
    const kinds = (/animated resources whose timelines cannot be frozen: ([^\n]*)/.exec(message) ?? ['', ''])[1].split(', ').map(x => x.split(' ')[0])
    expect(kinds.sort(), message).toEqual(['apng', 'gif', 'webp'])

    for (const where of ['/a.gif', '/a.png', '/a.webp']) {expect(message, where).toContain(where)}

    expect(message).not.toContain('/still.')
  }, 120_000)

  it('refuses a malformed SVG data URL used by a video poster, use, a cursor, shape-outside, filter, clip-path, a mask box image and a paint server', async () => {
    const message: string = (await svgLaunch()).error?.message ?? ''
    const shas = new Set(sentences(message).map(l => (/\(sha256 ([0-9a-f]{12})\)$/.exec(l) ?? [])[1]))

    for (const [name] of otherConsumers) {
      expect(shas.has(sha12(malformedFor(name))), `${name}: ${message}`).toBe(true)
    }
  }, 120_000)

  it('refuses a style-element SVG img on the prototype side, naming the data URL', async () => {
    const run = await launchPrototype(`<!doctype html><h1>Proto</h1><img alt="" src="${dataUri(SVG, unsupportedSvgs['style-element.svg'])}">`)
    expect(run.error.message).toMatch(/^unsupported content: the kit cannot prove that this SVG is static: data:image\/svg\+xml;base64,\S+… \(sha256 [0-9a-f]{12}\)$/)
  }, 120_000)

  // A malformed SVG data URL never produces a Network response; the shipped CLI must refuse it all the same, on both sides.
  const MALFORMED_SVG = `<svg ${SVG_NS}><rect></svg>`
  const MALFORMED_OTHER = `<svg ${SVG_NS}><g></svg>`

  const shippedCli = (args: string[]) => limited(() => new Promise<{ status: number | null, stderr: string, out: string }>(done => {
    const out = join(scratch, `out-${++seq}`)
    const child = spawn(process.execPath, [captureFile, ...args, '--out', out, '--run-id', 'r', '--candidate-digest', 'd'], { env: { ...process.env, CHROME_PATH: chromium } })

    let stderr = ''
    child.stderr.on('data', d => { stderr += d })
    child.on('close', status => done({ status, stderr, out }))
  }))

  const sentences = (stderr: string) => stderr.split('\n').filter(l => l.startsWith('unsupported content: the kit cannot prove that this SVG is static: '))
  const DATA_LINE = /^unsupported content: the kit cannot prove that this SVG is static: data:image\/svg\+xml \(sha256 ([0-9a-f]{12})\)$/

  // An SVG filter image is a rendered consumer of a data URL as well: feImage by href and by xlink:href.
  const feImageSvg = (attr: string, bytes: string) => `<svg width="64" height="64"><filter id="f${++seq}" x="0" y="0" width="64" height="64" filterUnits="userSpaceOnUse"><feImage ${attr}="${dataUri(SVG, bytes)}" width="64" height="64"/></filter><rect width="64" height="64" filter="url(#f${seq})"/></svg>`
  const MALFORMED_HREF = `<svg ${SVG_NS}><circle></svg>`
  const MALFORMED_XLINK = `<svg ${SVG_NS}><path></svg>`
  const sha12 = (bytes: string) => createHash('sha256').update(bytes).digest('hex').slice(0, 12)

  // An SVG document under a generic XML media type is rendered as an SVG by img and CSS images, so its style element is refused like any other.
  const XML_TYPES = ['text/xml', 'application/xml']
  const xmlStyled = (label: string) => svgDoc(`<style>rect{fill:rgb(0,255,0)}</style><title>${label}</title>${fill}`)

  // Plain with both consumers; behind a 70,000-character comment and behind a DOCTYPE entity holding "]>" as img.
  const XML_CASES: Array<[string, string[]]> = [['', ['img', 'background']], ['long-comment', ['img']], ['entity-doctype', ['img']]]

  const xmlConsumers = XML_TYPES.flatMap(type => XML_CASES.flatMap(([prolog, kinds]) => kinds.map(kind => {
    const svg = (PROLOGS[prolog] ?? '') + xmlStyled(`${type} ${prolog} ${kind}`)
    const uri = dataUri(type, svg)

    const html = kind === 'img' ? `<img alt="" src="${uri}">` : `<div style="width:64px;height:64px;background:url(${uri})"></div>`

    return { type, kind, svg, html }
  })))

  const xmlShas = (stderr: string) => sentences(stderr).map(l => (/\(sha256 ([0-9a-f]{12})\)$/.exec(l) ?? [])[1]).sort()

  // The same media types around a style-free SVG, captured on the ok page of the malformed-data test below: they must not be refused.
  const xmlStill = XML_TYPES.map(type => `<img alt="" src="${dataUri(type, stillSvg(type))}"><div style="width:64px;height:64px;background:url(${dataUri(type, GRADIENT)})"></div>`
    + Object.entries(PROLOGS).map(([name, prolog]) => `<img alt="" src="${dataUri(type, prolog + stillSvg(`${type} ${name}`))}">`).join('')).join('')

  // One shipped-CLI process per side serves both the malformed-data tests and the XML-labelled tests: every refusal of a page is reported together.
  const malformedCandidate = memo(() => {
    const site = dirOf({
      'index.html': `<!doctype html><style>img{display:block}div{width:64px;height:64px;background:url(data:image/svg+xml,${encodeURIComponent(MALFORMED_OTHER)})}</style><h1>Bad</h1>
<img alt="" src="${dataUri(SVG, MALFORMED_SVG)}"><div></div><img alt="" src="${dataUri(SVG, GRADIENT)}">${feImageSvg('href', MALFORMED_HREF)}${feImageSvg('xlink:href', MALFORMED_XLINK)}${feImageSvg('href', GRADIENT)}${xmlConsumers.map(c => c.html).join('')}`,
      'ok.html': `<!doctype html><style>body{margin:0}svg,img,div{display:block}</style>${feImageSvg('href', GRADIENT)}${feImageSvg('xlink:href', GRADIENT)}<h1>Ok</h1>${xmlStill}`,
    })

    return shippedCli(['--side', 'candidate', '--site', site, '--pages', '/ok /'])
  })

  const malformedPrototype = memo(() => {
    const file = join(scratch, `proto-${++seq}.html`)
    writeFileSync(file, `<!doctype html><h1>Proto</h1><img alt="" src="${dataUri(SVG, MALFORMED_SVG)}">${feImageSvg('href', MALFORMED_HREF)}${xmlConsumers.map(c => c.html).join('')}`)

    return shippedCli(['--side', 'prototype', '--prototype', file, '--pages', '/'])
  })

  const isXmlLine = (line: string) => / data:(?:text|application)\/xml[ ;]/.test(line)

  it('refuses a malformed SVG data URL used as an img, a CSS background or feImage (href and xlink:href) on the candidate side, and renders the valid static feImage, through the shipped CLI', async () => {
    const run = await malformedCandidate()
    expect(run.status, run.stderr).toBe(1)
    const all = sentences(run.stderr)
    expect(all, run.stderr).toHaveLength(4 + xmlConsumers.length)
    const lines = all.filter(l => !isXmlLine(l))
    expect(lines, run.stderr).toHaveLength(4)

    const shas = lines.map(l => (DATA_LINE.exec(l) ?? [])[1])
    expect(shas.sort(), run.stderr).toEqual([MALFORMED_SVG, MALFORMED_OTHER, MALFORMED_HREF, MALFORMED_XLINK].map(sha12).sort())
    expect(run.stderr).not.toContain(Buffer.from(MALFORMED_SVG).toString('base64'))

    for (const width of [1280, 375]) {
      expect(contractProblems(read(run.out, width, '/ok')), `${width}`).toEqual([])
      const image = decodePng(pngOf(run.out, width, '/ok'))

      for (const y of [32, 96]) {
        const [r, g, b] = image.at(32, y)
        expect([Math.abs(r - 126), g, Math.abs(b - 128)].every(d => d <= 3), `${width} y=${y}: ${[r, g, b]}`).toBe(true)
      }
    }
  }, 120_000)

  it('refuses a malformed SVG data URL used as an img or by feImage on the prototype side, through the shipped CLI', async () => {
    const run = await malformedPrototype()
    expect(run.status, run.stderr).toBe(1)
    const all = run.stderr.trim().split('\n')
    expect(all).toHaveLength(2 + xmlConsumers.length)
    const lines = all.filter(l => !isXmlLine(l))
    expect(lines).toHaveLength(2)

    for (const line of lines) {expect(line).toMatch(DATA_LINE)}
    expect(lines.map(l => (DATA_LINE.exec(l) ?? [])[1]).sort()).toEqual([MALFORMED_SVG, MALFORMED_HREF].map(sha12).sort())
  }, 120_000)

  it('refuses a style-element SVG labelled text/xml or application/xml as an img and a CSS background, also behind a 70,000-character comment or a DOCTYPE entity holding "]>" as an img, on the candidate side, through the shipped CLI', async () => {
    const run = await malformedCandidate()
    expect(run.status, run.stderr).toBe(1)
    const lines = sentences(run.stderr).filter(isXmlLine)
    expect(lines, run.stderr).toHaveLength(xmlConsumers.length)
    expect(xmlShas(lines.join('\n')), run.stderr).toEqual(xmlConsumers.map(c => sha12(c.svg)).sort())
  }, 120_000)

  it('refuses a style-element SVG labelled text/xml or application/xml as an img and a CSS background, also behind a 70,000-character comment or a DOCTYPE entity holding "]>" as an img, on the prototype side, through the shipped CLI', async () => {
    const run = await malformedPrototype()
    expect(run.status, run.stderr).toBe(1)
    const lines = sentences(run.stderr).filter(isXmlLine)
    expect(lines, run.stderr).toHaveLength(xmlConsumers.length)
    expect(xmlShas(lines.join('\n')), run.stderr).toEqual(xmlConsumers.map(c => sha12(c.svg)).sort())
  }, 120_000)
})
