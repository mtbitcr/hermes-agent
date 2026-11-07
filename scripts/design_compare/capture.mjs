#!/usr/bin/env node
import { execFileSync, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { basename, extname, join, resolve, sep } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

// Chromium 153.0.8010.0 as extracted from the npm package below; any other binary or hash is refused.
export const CHROMIUM_PIN = {
  package: '@sparticuz/chromium@153.0.0', version: '153.0.8010.0',
  sha256: '53a15d6c3a3d27dfb54c4ba60278b1683136f70cf1e67e989da7dfbd3d451ef0',
}
export const VIEWPORTS = [[1280, 800], [375, 812]]
export const MAX_PAGE_HEIGHT = 8000
export const CHROMIUM_SWITCHES = ['--headless=new', '--disable-gpu', '--disable-lcd-text', '--font-render-hinting=none',
  '--force-color-profile=srgb', '--hide-scrollbars', '--disable-features=PaintHolding', '--lang=en-US',
  '--force-webrtc-ip-handling-policy=disable_non_proxied_udp']
// Forced values: from a cascade layer declared first they also beat author ID-selector !important rules; inline !important is rewritten.
export const FORCED = { animation: 'none', transition: 'none', 'caret-color': 'transparent', 'scroll-behavior': 'auto',
  '-webkit-font-smoothing': 'antialiased', 'text-rendering': 'geometricPrecision' }
export const STYLE = `@layer dc-normalise{*,*::before,*::after{${Object.entries(FORCED).map(([k, v]) => `${k}:${v}!important`).join(';')}}}`
// Runs at document creation in every frame, before page scripts: installs the layer, adopts it into every shadow root (attachShadow hook, adoptedStyleSheets
// setter) and keeps author inline !important out of the forced properties. The layer cannot beat inline !important, so inline declarations are rewritten
// synchronously on the paths a page can use to set them (setProperty, cssText, the style property and attribute) and again before every getComputedStyle
// and computedStyleMap read. Any other path is caught by a MutationObserver (which the parser flushes before it runs the next script); if that late repair
// had to change an element the page had already read a computed style of, the capture is marked and refused, because the page may have seen the old value.
function earlyNormalise(forced, css) {
  const style = document.createElement('style')
  style.id = '__dc_style'
  style.dataset.early = document.readyState === 'loading' ? '1' : '0'
  style.textContent = css
  const sheet = new CSSStyleSheet()
  sheet.replaceSync(css)
  const watch = { childList: true, subtree: true, attributes: true, attributeFilter: ['style'] }
  const declaration = CSSStyleDeclaration.prototype
  const setProperty = declaration.setProperty, getPriority = declaration.getPropertyPriority
  const cssText = Object.getOwnPropertyDescriptor(declaration, 'cssText')
  const observed = new WeakSet()
  // Returns true when the declaration had to be rewritten.
  const fixDeclaration = s => {
    if (s.parentRule !== null || !Array.from(s).some(p => getPriority.call(s, p))) return false
    const before = cssText.get.call(s)
    for (const [n, v] of Object.entries(forced)) setProperty.call(s, n, v, 'important')
    return cssText.get.call(s) !== before
  }
  const fix = el => { try { return !!el.style && fixDeclaration(el.style) } catch { return false } }
  const late = el => { if (fix(el) && observed.has(el)) style.dataset.tainted = '1' }
  const adopt = r => {
    if (!r.adoptedStyleSheets.includes(sheet)) r.adoptedStyleSheets = [sheet, ...r.adoptedStyleSheets]
    mo.observe(r, watch)
  }
  const scan = n => {
    for (const el of n.nodeType === 1 ? [n, ...n.querySelectorAll('*')] : n.querySelectorAll('*')) {
      late(el)
      if (el.shadowRoot) { adopt(el.shadowRoot); scan(el.shadowRoot) }
    }
  }
  const put = () => document.documentElement && !style.isConnected && document.documentElement.appendChild(style)
  const mo = new MutationObserver(records => {
    put()
    for (const r of records) { if (r.type === 'attributes') late(r.target); else r.addedNodes.forEach(n => n.nodeType === 1 && scan(n)) }
  })
  mo.observe(document, watch)
  put()
  const attach = Element.prototype.attachShadow
  Element.prototype.attachShadow = function (init) { const r = attach.call(this, init); adopt(r); return r }
  const own = Object.getOwnPropertyDescriptor(ShadowRoot.prototype, 'adoptedStyleSheets')
  Object.defineProperty(ShadowRoot.prototype, 'adoptedStyleSheets', {
    ...own, set(v) { const all = Array.from(v); own.set.call(this, all.includes(sheet) ? all : [sheet, ...all]) },
  })
  declaration.setProperty = function setProperty_(...args) { const r = setProperty.apply(this, args); fixDeclaration(this); return r }
  Object.defineProperty(declaration, 'cssText', { ...cssText, set(v) { cssText.set.call(this, v); fixDeclaration(this) } })
  for (const proto of [HTMLElement, SVGElement, window.MathMLElement].filter(Boolean).map(c => c.prototype)) {
    const inline = Object.getOwnPropertyDescriptor(proto, 'style')
    if (inline && inline.set) Object.defineProperty(proto, 'style', { ...inline, set(v) { inline.set.call(this, v); fix(this) } })
  }
  for (const name of ['setAttribute', 'setAttributeNS']) {
    const original = Element.prototype[name]
    Element.prototype[name] = function (...args) {
      const r = original.apply(this, args)
      if (String(args[name === 'setAttribute' ? 0 : 1]).toLowerCase() === 'style') fix(this)
      return r
    }
  }
  const read = window.getComputedStyle
  window.getComputedStyle = function getComputedStyle_(el, ...rest) {
    if (el && el.nodeType === 1) { fix(el); observed.add(el) }
    return read.call(this, el, ...rest)
  }
  const map = Element.prototype.computedStyleMap
  if (map) Element.prototype.computedStyleMap = function () { fix(this); observed.add(this); return map.apply(this, arguments) }
  style.dataset.sync = '1'
}
const INJECT = `(${earlyNormalise})(${JSON.stringify(FORCED)}, ${JSON.stringify(STYLE)})`
const PROTOTYPE_PREFIX = '/__prototype__/'
const REFUSAL = 'unsupported content: the kit captures single documents only'
const TYPES = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.woff2': 'font/woff2', '.woff': 'font/woff' }
const dir = fileURLToPath(new URL('.', import.meta.url))
const sleep = ms => new Promise(r => setTimeout(r, ms))
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex')
const versionOf = text => (/\d+(\.\d+){3}/.exec(text) || [''])[0]

// onlyPath is set on the prototype side: then the prototype document itself is the only local URL allowed.
export function classifyRequest(url, origin, onlyPath = null) {
  try {
    const u = new URL(url)
    if (['data:', 'blob:', 'about:'].includes(u.protocol)) return 'allowed'
    return u.protocol === 'http:' && u.origin === origin && (onlyPath === null || u.pathname === onlyPath) ? 'allowed' : 'blocked'
  } catch {
    return 'blocked'
  }
}

export function assertSelfContained(blocked) {
  if (blocked.length) throw Object.assign(new Error('the chosen design file is not self-contained'), { blocked })
}

// Walks a pierced CDP DOM tree (shadow roots included, template content excluded); returns the host's node name.
export function findClosedShadowRoot(node) {
  const roots = node.shadowRoots || []
  if (roots.some(r => r.shadowRootType === 'closed')) return node.nodeName
  for (const kid of [...roots, ...(node.children || [])]) {
    const hit = findClosedShadowRoot(kid)
    if (hit) return hit
  }
  return null
}

// Rendered native timelines cannot be paused or rewound from the page, so a capture whose resources contain one is refused. Raster formats are judged from
// the response bytes (GIF/APNG/WebP/AVIF frames); SVG resources by the allowlist below. Both read the response, so http, data: and blob: resources work alike.
const u32 = (b, i) => b.readUInt32BE(i)

export function gifFrameCount(buf) {
  const b = Buffer.from(buf)
  if (b.length < 13 || !/^GIF8[79]a$/.test(b.toString('latin1', 0, 6))) return 0
  let i = 13 + (b[10] & 0x80 ? 3 * 2 ** ((b[10] & 7) + 1) : 0)
  let frames = 0
  const skipBlocks = at => { while (at < b.length && b[at]) at += b[at] + 1; return at + 1 }
  while (i < b.length) {
    if (b[i] === 0x21) i = skipBlocks(i + 2)
    else if (b[i] === 0x2c) {
      frames++
      i = b.length < i + 10 ? b.length : skipBlocks(i + 11 + (b[i + 9] & 0x80 ? 3 * 2 ** ((b[i + 9] & 7) + 1) : 0))
    } else break
  }
  return frames
}

// The frame count a PNG announces in an acTL chunk that precedes its image data (1 for an ordinary PNG, 0 when it is not a PNG).
export function apngFrameCount(buf) {
  const b = Buffer.from(buf)
  if (b.length < 8 || b.toString('latin1', 1, 4) !== 'PNG') return 0
  for (let i = 8; i + 8 <= b.length; i += 12 + u32(b, i)) {
    const type = b.toString('latin1', i + 4, i + 8)
    if (type === 'acTL') return i + 16 <= b.length ? u32(b, i + 8) : 1
    if (type === 'IDAT' || type === 'IEND') return 1
  }
  return 1
}

export function webpIsAnimated(buf) {
  const b = Buffer.from(buf)
  return b.length >= 21 && b.toString('latin1', 0, 4) === 'RIFF' && b.toString('latin1', 8, 12) === 'WEBP' && b.toString('latin1', 12, 16) === 'VP8X' && !!(b[20] & 2)
}

// The kind of animated raster image these bytes are, or null.
export function animatedResource(buf, mime = '') {
  const b = Buffer.from(buf)
  const type = String(mime).split(';')[0].trim().toLowerCase()
  if (gifFrameCount(b) > 1) return 'gif'
  if (apngFrameCount(b) > 1) return 'apng'
  if (webpIsAnimated(b)) return 'webp'
  if (b.toString('latin1', 4, 8) === 'ftyp' && b.toString('latin1', 8, Math.min(b.length, Math.max(u32(b, 0), 12))).includes('avis')) return 'avif'
  if (type === 'image/jxl' || (b[0] === 0xff && b[1] === 0x0a) || b.toString('hex', 0, 12) === '0000000c4a584c200d0a870a') return 'jxl'
  return null
}

const SVG_ELEMENTS = ['svg', 'g', 'defs', 'title', 'desc', 'path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'text', 'tspan',
  'linearGradient', 'radialGradient', 'stop', 'clipPath', 'mask', 'pattern']

// Runs in a blank browser page, never in the captured page: parses the COMPLETE bytes of a resource with the browser's own XML parser (so prologs, comments,
// doctypes, character references and entities are decoded by the browser, not guessed at) and says whether the resource may pass. An SVG passes only without
// processing instructions, unlisted elements, style attributes or on* attributes; nothing in such a document can start a timeline, and CSS is not parsed at all.
// Bytes that are not well-formed XML cannot be classified, so they are refused when `claimsXml` says the browser may still show them as an SVG or XML document.
const svgMayPass = (base64, allowed, claimsXml) => {
  const bytes = Uint8Array.from(atob(base64), c => c.charCodeAt(0))
  const encoding = bytes[0] === 0xff && bytes[1] === 0xfe || bytes[0] === 0x3c && bytes[1] === 0 ? 'utf-16le'
    : bytes[0] === 0xfe && bytes[1] === 0xff || bytes[0] === 0 && bytes[1] === 0x3c ? 'utf-16be' : 'utf-8'
  const doc = new DOMParser().parseFromString(new TextDecoder(encoding).decode(bytes), 'image/svg+xml')
  if (doc.getElementsByTagNameNS('*', 'parsererror').length) return !claimsXml
  if (!doc.documentElement || doc.documentElement.localName !== 'svg') return true
  const plain = node => {
    for (let n = node.firstChild; n; n = n.nextSibling) {
      if (n.nodeType === 7) return false
      if (n.nodeType !== 1) continue
      if (!allowed.includes(n.localName)) return false
      for (const a of n.attributes) {
        if ([a.name, a.localName].some(name => name.toLowerCase() === 'style' || name.toLowerCase().startsWith('on'))) return false
      }
      if (!plain(n)) return false
    }
    return true
  }
  return plain(doc)
}

const percentDecode = text => {
  const raw = Buffer.from(text, 'utf8')
  const out = []
  for (let i = 0; i < raw.length; i++) {
    const hex = raw.toString('latin1', i + 1, i + 3)
    if (raw[i] === 0x25 && /^[0-9a-f]{2}$/i.test(hex)) { out.push(parseInt(hex, 16)); i += 2 } else out.push(raw[i])
  }
  return Buffer.from(out)
}

const XML_TYPE = /^(?:(?:text|application)\/xml|[a-z0-9.+-]+\/[a-z0-9.+-]*\+xml)$/
const mediaType = mime => String(mime ?? '').split(';')[0].trim().toLowerCase()

// Whether the media type says the browser may show the bytes as an SVG image or an XML document.
export const claimsXml = mime => mediaType(mime) === 'image/svg+xml' || XML_TYPE.test(mediaType(mime))

// The bytes the browser would decode from a data: URL labelled image/svg+xml or by an XML media type (base64 or percent-encoded payload), or null for any other
// URL. What the bytes hold is for the browser to say.
export function svgDataUrlBytes(url) {
  const m = /^data:([^,]*),([\s\S]*)$/i.exec(String(url).trim())
  if (!m) return null
  const [type, ...params] = m[1].split(';').map(p => p.trim().toLowerCase())
  if (!claimsXml(type)) return null
  const raw = percentDecode(m[2].split('#')[0])
  return params.at(-1) === 'base64' ? Buffer.from(raw.toString('latin1').replace(/[\t\n\f\r ]/g, ''), 'base64') : raw
}

// The url("...") items of computed style strings; the browser serialises them double-quoted with only \" \\ and hex escapes, so no CSS is parsed here.
export function cssUrls(value) {
  const unescape = text => text.replace(/\\(?:([0-9a-f]{1,6}) ?|([\s\S]))/gi, (_, hex, ch) => (hex ? String.fromCodePoint(parseInt(hex, 16)) : ch))
  return Array.from(String(value ?? '').matchAll(/url\("((?:[^"\\]|\\[\s\S])*)"\)/g), m => unescape(m[1]))
}

// The URLs of a srcset attribute: a URL is a run of non-space characters (so a data URL keeps its own commas), and descriptors end at the next comma outside parentheses.
export function srcsetUrls(value) {
  const s = String(value ?? '')
  const urls = []
  for (let i = 0; i < s.length;) {
    while (i < s.length && /[\s,]/.test(s[i])) i++
    let j = i
    while (j < s.length && !/\s/.test(s[j])) j++
    let url = s.slice(i, j)
    const trailing = /,+$/.exec(url)
    if (trailing) url = url.slice(0, -trailing[0].length)
    else for (let depth = 0; j < s.length && (s[j] !== ',' || depth); j++) depth += s[j] === '(' ? 1 : s[j] === ')' ? -1 : 0
    if (url) urls.push(url)
    i = Math.max(j, i + 1)
  }
  return urls
}

// Each distinct SVG data resource among the raw strings the probe collected from the page, named by a short prefix (its sha256 is printed with it).
export function svgDataResources({ urls = [], srcsets = [], styles = [] }) {
  const found = new Map()
  for (const url of [...urls, ...srcsets.flatMap(srcsetUrls), ...styles.flatMap(cssUrls)]) {
    const bytes = svgDataUrlBytes(url)
    const hash = bytes && sha256(bytes)
    if (bytes && !found.has(hash)) found.set(hash, { url: `data:${mediaType(String(url).trim().slice(5).split(/[;,]/)[0])}`, sha256: hash, bytes })
  }
  return [...found.values()]
}

// The kit captures a page only when it is one document: a child frame in the CDP frame tree (readable or not, in this process or another) refuses the capture.
export function assertNoChildFrames(frameTree) {
  const child = (frameTree.childFrames || [])[0]
  if (child) throw new Error(`${REFUSAL}: found a frame ${child.frame.url || child.frame.id}`)
}

const EMBEDDING = new Set(['iframe', 'frame', 'fencedframe', 'object', 'embed'])

// Walks a pierced CDP DOM tree (the head, every child of html, open and closed shadow roots); an element that embeds another document refuses the capture.
export function assertNoEmbeddingElements(node) {
  const name = String(node.localName || '').toLowerCase()
  if (EMBEDDING.has(name)) throw new Error(`${REFUSAL}: found <${name}>`)
  for (const kid of [...(node.shadowRoots || []), ...(node.children || [])]) assertNoEmbeddingElements(kid)
}

const shortUrl = url => (url.length > 60 ? `${url.slice(0, 60)}…` : url)

export function assertNoAnimatedResources(found, unreadable, unprovable = []) {
  const problems = []
  if (found.length) {
    problems.push(`unsupported content: animated resources whose timelines cannot be frozen: ${found.map(f => `${f.kind} ${shortUrl(f.url)} (sha256 ${f.sha256.slice(0, 12)})`).join(', ')}`)
  }
  for (const u of unprovable) problems.push(`unsupported content: the kit cannot prove that this SVG is static: ${shortUrl(u.url)} (sha256 ${u.sha256.slice(0, 12)})`)
  if (unreadable.length) {
    problems.push(`unsupported content: an image or document resource could not be read to check it for animation: ${unreadable.map(u => shortUrl(u.url)).join(', ')}`)
  }
  if (problems.length) throw new Error(problems.join('\n'))
}

// Runs before any page script in every frame: the Fetch gate cannot see WebSocket handshakes, WebRTC/ICE or WebTransport, so refuse those at the API.
const guardSource = (origin, onlyPath) => `(() => {
  const classify = ${classifyRequest}, origin = ${JSON.stringify(origin)}, only = ${JSON.stringify(onlyPath)}
  const report = window.__dcBlock
  const abs = u => { try { return new URL(u, location.href).href } catch { return String(u) } }
  const ok = u => classify(abs(u), origin, only) === 'allowed'
  const refuseAll = urls => { urls.forEach(u => report(abs(u))); throw new DOMException('outside request refused', 'SecurityError') }
  const refuse = u => refuseAll([u])
  window.open = u => { report(abs(u || 'about:blank')); return null }
  const Source = window.EventSource
  window.EventSource = class extends Source { constructor(u, o) { if (!ok(u)) refuse(u); super(u, o) } }
  // Worker globals cannot be guarded from here, so no worker may start at all.
  for (const name of ['Worker', 'SharedWorker', 'WebSocket', 'WebSocketStream', 'WebTransport']) {
    if (name in window) window[name] = class { constructor(u) { refuse(u) } }
  }
  // ICE servers (STUN/TURN) and data channels never go through Fetch: refuse before construction and report every server named.
  const ice = c => { try { return c.iceServers.flatMap(s => [].concat(s.urls)).map(String) } catch { return [] } }
  for (const name of ['RTCPeerConnection', 'webkitRTCPeerConnection', 'RTCDataChannel']) {
    if (name in window) window[name] = class { constructor(c) { refuseAll(['webrtc:' + name, ...ice(c)]) } }
  }
  if (window.ServiceWorkerContainer) {
    ServiceWorkerContainer.prototype.register = function (u) { report(abs(u)); return Promise.reject(new DOMException('outside request refused', 'SecurityError')) }
  }
  const beacon = Navigator.prototype.sendBeacon
  Navigator.prototype.sendBeacon = function (u, d) { if (ok(u)) return beacon.call(this, u, d); report(abs(u)); return false }
  if (window.fetchLater) window.fetchLater = u => refuse(u && u.url || u)
})()`

// A binary is hashed once per process while its file identity (path, size, inode, mtime, ctime) stays the same; any change to the file hashes it again.
const hashed = new Map()
function binaryHash(binary) {
  let id
  try { const st = statSync(binary); id = [binary, st.size, st.ino, st.mtimeMs, st.ctimeMs].join('\0') } catch { throw new Error(`Chromium binary is not readable: ${binary}`) }
  if (!hashed.has(id)) {
    try { hashed.set(id, sha256(readFileSync(binary))) } catch { throw new Error(`Chromium binary is not readable: ${binary}`) }
  }
  return hashed.get(id)
}

export function verifyChromium(binary, pin, deps = {}) {
  if (!/^[0-9a-f]{64}$/.test(pin.sha256 || '')) throw new Error('Chromium binary hash is not pinned (missing hash): refusing to start')
  if (binaryHash(binary) !== pin.sha256) throw new Error('Chromium binary SHA-256 does not match the pin: refusing to start')
  const version = versionOf((deps.version || (b => execFileSync(b, ['--version'], { encoding: 'utf8' })))(binary))
  if (version !== pin.version) throw new Error(`Chromium version ${version || 'unknown'} is not the pinned version ${pin.version}`)
  return version
}

export function buildBundle(i) {
  return {
    schema: 'design_compare_bundle_v1', kit_sha256: i.kitSha256, chromium_version: i.chromiumVersion, run_id: i.runId,
    candidate_digest: i.candidateDigest, page: i.page, width: i.width, page_height: i.pageHeight, image: i.image,
    blocked_requests: [...new Set(i.blockedRequests)].sort(), masks: i.measurement.masks, elements: i.measurement.elements,
  }
}

export const pageSlug = route => `${route.replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '') || 'index'}-${sha256(route).slice(0, 8)}`

export async function startServer({ siteDir, prototypeFile }) {
  const isFile = p => existsSync(p) && statSync(p).isFile()
  const server = createServer((req, res) => {
    let path
    try { path = decodeURIComponent(new URL(req.url, 'http://local').pathname) } catch { path = null }
    let file
    if (prototypeFile && path === PROTOTYPE_PREFIX + basename(prototypeFile)) file = prototypeFile
    else if (siteDir && path !== null) {
      const root = resolve(siteDir)
      const c = resolve(root, `.${path}`)
      if (c === root || c.startsWith(root + sep)) file = [c, join(c, 'index.html'), `${c}.html`].find(isFile)
    }
    if (!file) { res.writeHead(404).end(); return }
    res.writeHead(200, { 'content-type': TYPES[extname(file)] || 'application/octet-stream' }).end(readFileSync(file))
  })
  await new Promise(r => server.listen(0, '127.0.0.1', r))
  return { origin: `http://127.0.0.1:${server.address().port}`, close: () => new Promise(r => server.close(r)) }
}

async function launch(chromium) {
  const profile = mkdtempSync(join(tmpdir(), 'dc-chrome-'))
  const args = [...CHROMIUM_SWITCHES, '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run',
    ...(process.getuid && process.getuid() === 0 ? ['--no-sandbox'] : []), 'about:blank']
  const child = spawn(chromium, args, { stdio: ['ignore', 'ignore', 'pipe'] })
  const wsUrl = await new Promise((ok, no) => {
    let seen = ''
    const timer = setTimeout(() => no(new Error('Chromium did not start')), 30000)
    child.stderr.on('data', d => {
      seen += d
      const m = /DevTools listening on (ws:\/\/\S+)/.exec(seen)
      if (m) { clearTimeout(timer); ok(m[1]) }
    })
    child.once('exit', () => { clearTimeout(timer); no(new Error('Chromium exited early')) })
  })
  const ws = new WebSocket(wsUrl)
  await new Promise((ok, no) => { ws.onopen = ok; ws.onerror = () => no(new Error('cannot connect to Chromium')) })
  let id = 0
  const pending = new Map()
  const listeners = []
  ws.onmessage = ev => {
    const m = JSON.parse(ev.data)
    if (!m.id) { listeners.forEach(l => l(m)); return }
    const p = pending.get(m.id)
    pending.delete(m.id)
    if (m.error) p.no(new Error(`${p.method}: ${m.error.message}`))
    else p.ok(m.result)
  }
  ws.onclose = () => pending.forEach(p => p.no(new Error('Chromium connection closed')))
  return {
    send: (method, params = {}, sessionId) => new Promise((ok, no) => {
      pending.set(++id, { ok, no, method })
      ws.send(JSON.stringify({ id, method, params, sessionId }))
    }),
    on: l => listeners.push(l),
    async close() {
      ws.onclose = null
      ws.close()
      child.kill('SIGKILL')
      if (child.exitCode === null) await new Promise(r => child.once('exit', r))
      rmSync(profile, { force: true, recursive: true, maxRetries: 3 })
    },
  }
}

async function openTarget(browser) {
  const { targetId } = await browser.send('Target.createTarget', { url: 'about:blank' })
  const { sessionId } = await browser.send('Target.attachToTarget', { targetId, flatten: true })
  const send = (method, params) => browser.send(method, params, sessionId)
  const evaluate = async (expression, contextId) => {
    const r = await send('Runtime.evaluate', { expression, contextId, returnByValue: true, awaitPromise: true })
    if (r.exceptionDetails) throw new Error((r.exceptionDetails.exception || {}).description || r.exceptionDetails.text)
    return r.result.value
  }
  return { sessionId, send, evaluate, close: () => browser.send('Target.closeTarget', { targetId }) }
}

const frames = 'new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))'

const AUTO_ATTACH = { autoAttach: true, waitForDebuggerOnStart: true, flatten: true }

async function capturePage(browser, stitcher, probeSource, { url, origin, onlyPath, width, height }) {
  const t = await openTarget(browser)
  const blocked = []
  const leaked = []
  const open = new Set()
  const responses = new Map()
  const inspections = new Set()
  const animated = []
  const unprovable = []
  const unreadable = []
  const source = `${INJECT};${guardSource(origin, onlyPath)}`
  let lastActivity = Date.now()
  let onLoad
  const loaded = new Promise(r => { onLoad = r })
  // A frame that Chromium attaches as its own target is never resumed, so it never makes a request and never finishes loading: it ends the wait for the load.
  let embedded = null
  let onEmbedded
  const embeddedEarly = new Promise((_, no) => { onEmbedded = no })
  embeddedEarly.catch(() => {})
  // The gates of the capture page, applied while the session is still paused at start.
  const gate = async send => {
    for (const m of ['Page.enable', 'Network.enable', 'Runtime.enable']) await send(m)
    await send('Network.setCacheDisabled', { cacheDisabled: true })
    await send('Runtime.addBinding', { name: '__dcBlock' })
    await send('Fetch.enable', { patterns: [{ urlPattern: '*' }] })
    await send('Target.setAutoAttach', AUTO_ATTACH)
    await send('Page.addScriptToEvaluateOnNewDocument', { source })
  }
  browser.on(m => {
    if (m.method === 'Target.attachedToTarget') {
      const { targetInfo: info, waitingForDebugger } = m.params
      if (!waitingForDebugger) return
      if (m.sessionId === undefined) {
        // A new top-level target is a popup: never resumed, so it never makes a request.
        blocked.push(info.url || 'about:blank')
        browser.send('Target.closeTarget', { targetId: info.targetId }).catch(() => {})
      } else if (m.sessionId === t.sessionId && info.type === 'iframe') {
        embedded ??= new Error(`${REFUSAL}: found a frame ${info.url || info.targetId}`)
        onEmbedded(embedded)
      } else blocked.push(info.url || info.type)
      return
    }
    if (m.sessionId !== t.sessionId) return
    const { requestId, request } = m.params
    const send = (method, params) => browser.send(method, params, m.sessionId).catch(() => {})
    if (m.method === 'Fetch.requestPaused' || m.method === 'Runtime.bindingCalled' || m.method.startsWith('Network.')) lastActivity = Date.now()
    if (m.method === 'Fetch.requestPaused') {
      if (classifyRequest(request.url, origin, onlyPath) === 'blocked') {
        blocked.push(request.url)
        send('Fetch.failRequest', { requestId, errorReason: 'BlockedByClient' })
      } else send('Fetch.continueRequest', { requestId })
    } else if (m.method === 'Runtime.bindingCalled') blocked.push(m.params.payload)
    else if (m.method === 'Network.webSocketCreated') { blocked.push(m.params.url); leaked.push(m.params.url) }
    else if (m.method === 'Network.requestWillBeSent') open.add(requestId)
    else if (m.method === 'Network.responseReceived') responses.set(`${m.sessionId}:${requestId}`, { ...m.params.response, type: m.params.type })
    else if (m.method === 'Network.loadingFinished' || m.method === 'Network.loadingFailed') {
      open.delete(requestId)
      const key = `${m.sessionId}:${requestId}`
      const response = responses.get(key)
      responses.delete(key)
      if (m.method === 'Network.loadingFinished' && response && ['Image', 'Document', 'Other'].includes(response.type)) {
        const check = (async () => {
          const { body, base64Encoded } = await browser.send('Network.getResponseBody', { requestId }, m.sessionId)
          const bytes = Buffer.from(body, base64Encoded ? 'base64' : 'utf8')
          const kind = animatedResource(bytes, response.mimeType)
          if (kind) {
            if (!animated.some(a => a.url === response.url && a.kind === kind)) animated.push({ kind, url: response.url, sha256: sha256(bytes) })
          } else {
            const plain = await stitcher.evaluate(`(${svgMayPass})(${JSON.stringify(bytes.toString('base64'))}, ${JSON.stringify(SVG_ELEMENTS)}, ${claimsXml(response.mimeType)})`)
            if (!plain && !unprovable.some(u => u.url === response.url && u.sha256 === sha256(bytes))) unprovable.push({ url: response.url, sha256: sha256(bytes) })
          }
        })().catch(() => { unreadable.push({ url: response.url }) })
        inspections.add(check)
        check.then(() => inspections.delete(check))
      }
    }
    else if (m.method === 'Page.loadEventFired') onLoad()
  })
  try {
    // Runtime.enable makes Chromium create the script context at document start; otherwise a scriptless page runs the injected script after load.
    await browser.send('Target.setAutoAttach', AUTO_ATTACH)
    await gate(t.send)
    await t.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false })
    await t.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] })
    await t.send('Emulation.setTimezoneOverride', { timezoneId: 'UTC' })
    await t.send('Emulation.setLocaleOverride', { locale: 'en-US' })
    const nav = await t.send('Page.navigate', { url })
    if (nav.errorText) throw new Error(`cannot load ${url}: ${nav.errorText}`)
    await Promise.race([loaded, embeddedEarly, new Promise((_, no) => setTimeout(() => no(new Error(`timed out loading ${url}`)), 30000).unref())])
    const { frameTree } = await t.send('Page.getFrameTree')
    const world = await t.send('Page.createIsolatedWorld', { frameId: frameTree.frame.id, worldName: 'design_compare' })
    const run = expr => t.evaluate(expr, world.executionContextId)
    const probe = `(${probeSource}\n)`
    // The DOM and the frame tree are both read each time, so a frame that cannot be read, or is attached out of process, is refused as well.
    const assertSingleDocument = async () => {
      if (embedded) throw embedded
      assertNoEmbeddingElements((await t.send('DOM.getDocument', { depth: -1, pierce: true })).root)
      assertNoChildFrames((await t.send('Page.getFrameTree')).frameTree)
    }
    await assertSingleDocument()
    if (!await run("(() => { const s = document.getElementById('__dc_style'); return !!s && s.dataset.early === '1' })()")) {
      throw new Error(`the motion-neutralising stylesheet was not installed before page scripts ran for ${url}`)
    }
    // A data: image that fails to decode never produces a Network response, so the data URLs the page references are read from the page itself.
    const dataSeen = new Set()
    const checkDataSvgs = async () => {
      for (const r of svgDataResources(await run(`${probe}.dataUrls()`))) {
        if (dataSeen.has(r.sha256)) continue
        dataSeen.add(r.sha256)
        if (unprovable.some(u => u.sha256 === r.sha256 && u.url.startsWith('data:'))) continue
        try {
          const plain = await stitcher.evaluate(`(${svgMayPass})(${JSON.stringify(r.bytes.toString('base64'))}, ${JSON.stringify(SVG_ELEMENTS)}, true)`)
          if (!plain) unprovable.push({ url: r.url, sha256: r.sha256 })
        } catch { unreadable.push({ url: r.url }) }
      }
    }
    await run(`${probe}.toTop()`)
    await run('document.fonts.ready.then(() => true)')
    const quiet = async (settled = true) => {
      const deadline = Date.now() + 30000
      while (open.size || (settled && Date.now() - lastActivity < 500)) {
        if (Date.now() > deadline) throw new Error(`network never went quiet for ${url}`)
        await sleep(50)
      }
      while (inspections.size) await Promise.all([...inspections])
      await checkDataSvgs()
      assertNoAnimatedResources(animated, unreadable, unprovable)
    }
    await quiet()
    const tree = (await t.send('DOM.getDocument', { depth: -1, pierce: true })).root
    const closed = findClosedShadowRoot(tree)
    if (closed) throw new Error(`unsupported content: a closed shadow root under <${closed.toLowerCase()}> cannot be measured`)
    await run(`${probe}.toTop()`)
    await run(`${probe}.prepare(${JSON.stringify(FORCED)})`)
    await run(`${probe}.fontsReady()`)
    await run(frames)
    await run(`${probe}.settle()`)
    await assertSingleDocument()
    const measurement = await run(`${probe}.measure()`)
    const pageHeight = Math.min(measurement.pageHeight, MAX_PAGE_HEIGHT)
    const tiles = []
    for (let y = 0, n = 0; y < pageHeight; y += height, n++) {
      const top = Math.min(y, Math.max(pageHeight - height, 0))
      await run(`(async () => { const p = ${probe}\n; ${n === 0 ? 'p.toTop()' : `window.scrollTo(0, ${top})`}; p.freeze(); await ${frames}; ${n === 0 ? 'p.atTop()' : ''} })()`)
      if (n === 1) await run(`${probe}.hidePinned()`)
      await assertSingleDocument()
      tiles.push({ top, data: (await t.send('Page.captureScreenshot', { format: 'png' })).data })
    }
    await quiet(false)
    if (leaked.length) throw new Error(`outside WebSocket handshake observed: ${leaked.join(' ')}`)
    const png = await stitcher.evaluate(`(async (tiles, w, h) => {
      const c = document.createElement('canvas'); c.width = w; c.height = h
      const ctx = c.getContext('2d')
      let covered = 0
      for (const t of tiles) {
        const img = new Image(); img.src = 'data:image/png;base64,' + t.data; await img.decode()
        const skip = covered - t.top
        ctx.drawImage(img, 0, skip, img.width, img.height - skip, 0, covered, img.width, img.height - skip)
        covered = t.top + img.height
      }
      return c.toDataURL('image/png').split(',')[1]
    })(${JSON.stringify(tiles)}, ${width}, ${pageHeight})`)
    return { blocked, measurement, pageHeight, png: Buffer.from(png, 'base64') }
  } finally {
    await browser.send('Target.setAutoAttach', { ...AUTO_ATTACH, autoAttach: false }).catch(() => {})
    await t.close().catch(() => {})
  }
}

export async function capture({ side, siteDir, prototypeFile, pages, outDir, runId, candidateDigest, chromium, pin = CHROMIUM_PIN }) {
  if (side !== 'candidate' && side !== 'prototype') throw new Error('side must be candidate or prototype')
  const chromiumVersion = verifyChromium(chromium, pin)
  const probeSource = readFileSync(join(dir, 'probe.js'), 'utf8')
  const kitSha256 = sha256(Buffer.concat([readFileSync(join(dir, 'capture.mjs')), Buffer.from(probeSource)]))
  const targets = pages.flatMap(route => VIEWPORTS.flatMap(([width]) => ['png', 'json'].map(ext => join(outDir, `${pageSlug(route)}-${width}.${ext}`))))
  const taken = targets.find((f, i) => existsSync(f) || targets.indexOf(f) !== i)
  if (taken) throw new Error(`refusing to overwrite ${taken}`)
  mkdirSync(outDir, { recursive: true })
  const server = await startServer({ siteDir: side === 'candidate' ? siteDir : undefined, prototypeFile })
  const browser = await launch(chromium)
  const bundles = []
  try {
    const stitcher = await openTarget(browser)
    for (const route of pages) {
      const onlyPath = side === 'prototype' ? new URL(`${PROTOTYPE_PREFIX}${encodeURIComponent(basename(prototypeFile))}`, server.origin).pathname : null
      const url = onlyPath === null ? `${server.origin}${route}` : `${server.origin}${onlyPath}`
      for (const [width, height] of VIEWPORTS) {
        const shot = await capturePage(browser, stitcher, probeSource, { url, origin: server.origin, onlyPath, width, height })
        if (side === 'prototype') assertSelfContained(shot.blocked)
        const name = `${pageSlug(route)}-${width}`
        const bundle = buildBundle({
          kitSha256, chromiumVersion, runId, candidateDigest, page: route, width, pageHeight: shot.pageHeight, image: `${name}.png`,
          blockedRequests: shot.blocked, measurement: shot.measurement,
        })
        writeFileSync(join(outDir, `${name}.png`), shot.png, { flag: 'wx' })
        writeFileSync(join(outDir, `${name}.json`), `${JSON.stringify(bundle, null, 2)}\n`, { flag: 'wx' })
        bundles.push(bundle)
      }
    }
  } finally {
    await browser.close()
    await server.close()
  }
  return bundles
}

async function main() {
  const { values: v } = parseArgs({ options: {
    side: { type: 'string' }, site: { type: 'string' }, prototype: { type: 'string' }, pages: { type: 'string' },
    out: { type: 'string' }, 'run-id': { type: 'string' }, 'candidate-digest': { type: 'string' }, chromium: { type: 'string' },
  } })
  const chromium = v.chromium || process.env.CHROME_PATH
  if (!v.side || !v.pages || !v.out || !v['run-id'] || !v['candidate-digest'] || !chromium) {
    throw new Error('usage: capture.mjs --side candidate|prototype [--site DIR] [--prototype FILE] --pages "/ /x" --out DIR '
      + '--run-id ID --candidate-digest DIGEST [--chromium PATH or $CHROME_PATH]')
  }
  const bundles = await capture({
    side: v.side, siteDir: v.site, prototypeFile: v.prototype, pages: v.pages.split(/[\s,]+/).filter(Boolean), outDir: v.out,
    runId: v['run-id'], candidateDigest: v['candidate-digest'], chromium,
  })
  for (const b of bundles) console.log(`${b.page} ${b.width} ${b.image} blocked=${b.blocked_requests.length}`)
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch(e => { console.error(e.message, ...(e.blocked || []).map(u => `\n  requested: ${u}`)); process.exit(1) })
}
