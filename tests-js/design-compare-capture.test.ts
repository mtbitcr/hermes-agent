import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
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

  it('continues the landmark chain through shadow hosts and frame elements', () => {
    const inShadow = el('h2', {}, 'Inner')
    const host = el('div', {})
    el('body', {}, el('main', {}, host))
    inShadow.getRootNode = () => ({ host })
    const frame = el('iframe', {})
    el('body', {}, el('section', {}, frame))
    const inFrame = el('h3', {}, 'Framed')
    inFrame.ownerDocument = { defaultView: { frameElement: frame } }
    expect(probe.assignKeys([inShadow, inFrame])).toEqual(['main/h2:"inner"#1', 'section/h3:"framed"#1'])
  })

  it('prefers an explicit role to the tag name', () => {
    const d = el('div', { role: 'alert' }, 'Saved')
    el('body', {}, d)
    expect(probe.assignKeys([d])).toEqual(['/alert:"saved"#1'])
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
    const { settle } = runInNewContext(probeSource, { window: {}, scrollX: 0, scrollY: 0, document: { createElement: () => ({}), getAnimations: live, body: { querySelectorAll: () => [] } } })
    settle()

    return list.map(a => a.calls)
  }

  it('finishes finite animations, cancels infinite ones and fails when one is still running', () => {
    expect(settleWith(animation(500), animation(Infinity), animation(500, true))).toEqual([['finish'], ['cancel'], ['finish', 'cancel']])
    expect(() => settleWith(animation(Infinity, false, true))).toThrow('still running')
  })
})

const SHA = 'a'.repeat(64)

// PROVISIONAL fixture of design_compare_bundle_v1, derived from the wording of plan Q2/Q3 by this kit.
// C1 (task t_d7a1493a) had not run when this was written, so its schema constants did not exist and this is NOT
// a copy of them. Compare this fixture with C1's actual constants before C4 uses both.
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
  it('matches the provisional design_compare_bundle_v1 fixture', () => {
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
<script>document.getElementById('wa').animate([{transform:'translateX(0px)'},{transform:'translateX(300px)'}],{duration:${ms},iterations:Infinity})</script>`

  let motionRuns: Promise<[Awaited<ReturnType<typeof shoot>>, Awaited<ReturnType<typeof shoot>>]> | undefined
  const motion = () => (motionRuns ??= Promise.all([shoot({ 'index.html': motionPage(1000, 'antialiased') }), shoot({ 'index.html': motionPage(3700, 'subpixel-antialiased') })]))

  it('gives identical bundles and images despite CSS animation, infinite Web Animations API motion and font smoothing', async () => {
    const [a, b] = await motion()
    sameFiles(a.outDir, b.outDir)
    const bundle = JSON.parse(readFileSync(join(a.outDir, `${kit.pageSlug('/')}-1280.json`), 'utf8'))
    expect(conforms(bundle, BUNDLE_V1.fields)).toEqual([])
    expect(bundle.elements.find((e: any) => e.box.width === 80 && e.box.height === 60).box.x).toBe(0)
    expect(bundle.chromium_version).toBe('153.0.8010.0')
  }, 120_000)

  it('excludes elements under an opacity-zero ancestor and keeps visible controls', async () => {
    const keys = (await motion())[0].bundles[0].elements.map((e: any) => e.key)
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

    expect(byText(s10, 'visible shadow text').type.size).toBe(10)
    expect(byText(s40, 'visible shadow text').type.size).toBe(40)
    expect(byText(f10, 'frame text').type.size).toBe(10)
    expect(byText(f40, 'frame text').type.size).toBe(40)
    expect(JSON.stringify(s10)).not.toBe(JSON.stringify(s40))
    expect(JSON.stringify(f10)).not.toBe(JSON.stringify(f40))
    expect(byText(f10, 'frame text').box.x).toBeGreaterThanOrEqual(40)
    expect(byText(f10, 'frame text').box.y).toBeGreaterThan(60)
    expect(s10.masks).toContainEqual(expect.objectContaining({ width: 50, height: 20 }))
    expect(byText(f10, 'pin').pinned).toBe(true)
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

  it('enforces the rendering values against ID-selector and inline !important declarations, and fails when it cannot', async () => {
    const forced = '-webkit-font-smoothing:subpixel-antialiased!important;text-rendering:optimizeSpeed!important'

    const page = (script: string) => `<style>#a{${forced}}</style><h1 id=a>AVAVATAR</h1><p id=b style="${forced}">Inline</p><p id=ra></p><p id=rb></p>
<script>const show = () => { for (const [s, d] of [[a, ra], [b, rb]]) { const c = getComputedStyle(s); d.textContent = c.webkitFontSmoothing + ' / ' + c.textRendering } ${script} requestAnimationFrame(show) }; show()</script>`

    const { outDir } = await shoot({ 'index.html': page('') })

    for (const width of [1280, 375]) {
      const shown = read(outDir, width).elements.filter((e: any) => e.text === 'antialiased / geometricprecision')
      expect(shown, `${width}`).toHaveLength(2)
    }

    const err: any = await shoot({ 'index.html': page(`b.style.setProperty('-webkit-font-smoothing', 'subpixel-antialiased', 'important');`) }).catch((e: unknown) => e)
    expect(err.message).toContain('rendering normalisation is not effective')
  }, 120_000)
})
