#!/usr/bin/env node
// Compose one picture per design option from the capture kit's two widths, in the pinned browser.
//
//   node compose_offer.mjs --kit KIT --chromium BINARY --out DIR --option option-1.html CAPTURES1 --option option-2.html CAPTURES2
//
// For each option-<n>.html and the folder the capture kit wrote for it, the 1280 px capture goes left and the 375 px capture right on
// one canvas, labelled, at most 4 MiB: option-<n>-1280-375.png (or .jpg), plus design-offer-manifest.json that ties every picture to the
// page it shows. Everything that needs no pixels runs in Node (PNG sizes come from the IHDR chunk); the drawing runs in the pinned
// Chromium of KIT/capture.mjs, whose binary is verified first. Node built-ins only. Nothing from a bundle or a page is printed.
import { spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

export const MAX_PICTURE_BYTES = 4 * 1024 * 1024
export const MIN_OPTIONS = 2
export const MAX_OPTIONS = 3
export const WIDTHS = [1280, 375]
export const MANIFEST = 'design-offer-manifest.json'
export const COMPOSER = 'compose_offer.mjs'
export const BUNDLE_SCHEMA = 'design_compare_bundle_v1'
export const PAGE_NAME = /^option-([1-3])\.html$/
const SHA256 = /^[0-9a-f]{64}$/

export const MARGIN = 40
export const HEADER = 72
export const LABEL_SIZE = 32
export const BACKGROUND = [240, 240, 240]
export const LABEL_COLOR = [20, 20, 20]
export const JPEG_QUALITIES = [0.85, 0.7, 0.55, 0.4]
export const SCALE_STEP = 0.8
export const MIN_SCALE = 0.2
export const JPEG_TOLERANCE = 24
export const SAMPLE_GRID = 8

export class OfferError extends Error {}

const sha256 = data => createHash('sha256').update(data).digest('hex')

// A PNG starts with its signature and an IHDR chunk of 13 bytes: width and height are the big-endian words at bytes 16-23.
export function pngSize(buf) {
  const signature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
  if (buf.length < 24 || signature.some((b, i) => buf[i] !== b) || buf.readUInt32BE(8) !== 13 || buf.toString('latin1', 12, 16) !== 'IHDR') {
    throw new OfferError('not a PNG picture')
  }
  return { width: buf.readUInt32BE(16), height: buf.readUInt32BE(20) }
}

// Returns the options as { page, folder, n }, sorted by n; refuses anything but two or three pages option-1.html .. option-3.html with no gaps.
export function checkOptions(options) {
  if (options.length < MIN_OPTIONS || options.length > MAX_OPTIONS) throw new OfferError(`an offer needs two or three options, got ${options.length}`)
  const numbered = options.map(([page, folder]) => {
    const match = PAGE_NAME.exec(basename(page))
    if (!match) throw new OfferError(`${basename(page)}: a page must be named option-<n>.html with n from 1 to 3`)
    return { page, folder, n: Number(match[1]) }
  })
  numbered.sort((a, b) => a.n - b.n)
  if (numbered.some((o, i) => o.n !== i + 1)) throw new OfferError('the options must be numbered 1, 2 and 3 in order, with no gaps or repeats')
  return numbered
}

const isRegularFile = path => { try { const s = lstatSync(path); return s.isFile() && !s.isSymbolicLink() } catch { return false } }

export function readCapture(folder, width) {
  const name = basename(folder)
  const found = readdirSync(folder).filter(f => f.endsWith(`-${width}.json`)).sort()
  if (found.length !== 1) throw new OfferError(`${name}: expected exactly one ${width} px bundle, found ${found.length}`)
  let bundle
  try { bundle = JSON.parse(readFileSync(join(folder, found[0]), 'utf8')) } catch { throw new OfferError(`${name}: the ${width} px bundle cannot be read`) }
  if (!bundle || typeof bundle !== 'object' || bundle.schema !== BUNDLE_SCHEMA) throw new OfferError(`${name}: the ${width} px bundle is not a capture bundle`)
  if (bundle.width !== width) throw new OfferError(`${name}: the ${width} px bundle names another width`)
  if (!Array.isArray(bundle.blocked_requests) || bundle.blocked_requests.length) throw new OfferError(`${name}: the ${width} px capture blocked a request; refused`)
  for (const key of ['kit_sha256', 'chromium_version']) {
    if (typeof bundle[key] !== 'string' || !bundle[key]) throw new OfferError(`${name}: the ${width} px bundle has no ${key}`)
  }
  if (!SHA256.test(bundle.kit_sha256)) throw new OfferError(`${name}: the ${width} px bundle has a malformed kit_sha256`)
  const image = bundle.image
  if (typeof image !== 'string' || image !== basename(image) || !image.endsWith(`-${width}.png`)) {
    throw new OfferError(`${name}: the ${width} px bundle names a picture outside its folder`)
  }
  const path = join(folder, image)
  if (!isRegularFile(path)) throw new OfferError(`${name}: the ${width} px picture ${image} is missing`)
  const data = readFileSync(path)
  let size
  try { size = pngSize(data) } catch { throw new OfferError(`${name}: ${image} is not a PNG`) }
  if (size.width !== width || !Number.isInteger(bundle.page_height) || size.height !== bundle.page_height) {
    throw new OfferError(`${name}: ${image} does not match its bundle (width and page height)`)
  }
  return { width, file: image, sha256: sha256(data), data, height: size.height, kit_sha256: bundle.kit_sha256, chromium_version: bundle.chromium_version }
}

export function sharedKit(captures) {
  const kits = new Set(captures.map(c => c.kit_sha256))
  const versions = new Set(captures.map(c => c.chromium_version))
  if (kits.size !== 1) throw new OfferError('the bundles name different capture kit hashes')
  if (versions.size !== 1) throw new OfferError('the bundles name different Chromium versions')
  return { kit_sha256: [...kits][0], chromium_version: [...versions][0] }
}

// Sizes and places of one picture at one scale: labels above, the 1280 px capture left, the 375 px capture right.
export function layout(left, right, scale = 1) {
  const scaled = n => Math.max(1, Math.round(n * scale))
  const margin = Math.round(MARGIN * scale)
  const header = Math.round(HEADER * scale)
  const l = { x: margin, y: header, w: scaled(left.width), h: scaled(left.height) }
  const r = { x: margin * 2 + l.w, y: header, w: scaled(right.width), h: scaled(right.height) }
  const labelSize = Math.max(8, Math.round(LABEL_SIZE * scale))
  return {
    scale, width: margin * 3 + l.w + r.w, height: header + margin + Math.max(l.h, r.h), left: l, right: r, labelSize,
    labels: [{ text: `${WIDTHS[0]} px`, x: l.x, y: Math.floor(header / 2) }, { text: `${WIDTHS[1]} px`, x: r.x, y: Math.floor(header / 2) }],
  }
}

// At each scale: lossless PNG first, then JPEG from quality 0.85 down to 0.40; shrink evenly by 0.8 only when none fits.
export function encodingLadder() {
  const ladder = []
  for (let scale = 1; scale >= MIN_SCALE; scale *= SCALE_STEP) {
    ladder.push({ scale, mime: 'image/png' }, ...JPEG_QUALITIES.map(quality => ({ scale, mime: 'image/jpeg', quality })))
  }
  return ladder
}

// encode(step) returns the picture bytes, or null when they are known to exceed the limit. accept(data, step) returns null when the picture
// passes the output self-check, else the reason it does not. The first picture within the limit that also passes wins.
export async function chooseEncoding(encode, max = MAX_PICTURE_BYTES, accept = async () => null) {
  let reason = 'a picture cannot be made small enough for the offer tool (4 MiB)'
  for (const step of encodingLadder()) {
    const data = await encode(step)
    if (!data || data.length > max) continue
    const failure = await accept(data, step)
    if (failure === null) return { data, step, suffix: step.mime === 'image/png' ? '.png' : '.jpg' }
    reason = failure
  }
  throw new OfferError(reason)
}

// Centres of an even grid of cells inside one pane.
export function samplePoints(pane, grid = SAMPLE_GRID) {
  const points = []
  for (let row = 0; row < grid; row++) {
    for (let col = 0; col < grid; col++) {
      points.push({ x: pane.x + Math.min(pane.w - 1, Math.floor(((col + 0.5) * pane.w) / grid)), y: pane.y + Math.min(pane.h - 1, Math.floor(((row + 0.5) * pane.h) / grid)) })
    }
  }
  return points
}

export const toleranceFor = mime => (mime === 'image/png' ? 0 : JPEG_TOLERANCE)

export function pictureKind(data) {
  if (data.length > 8 && data.subarray(0, 8).equals(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]))) return 'image/png'
  if (data.length > 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) return 'image/jpeg'
  return null
}

export function buildManifest({ kit_sha256, chromium_version, composerChromiumVersion, entries }) {
  return { kit_sha256, chromium_version, composer: COMPOSER, composer_chromium_version: composerChromiumVersion, options: entries }
}

// Refuses to overwrite anything; creates every file exclusively and removes what it created if a later one fails.
export function writeOutputs(outDir, outputs) {
  const taken = outputs.filter(([name]) => existsSync(join(outDir, name))).map(([name]) => name)
  if (taken.length) throw new OfferError(`refusing to overwrite ${taken.join(', ')}`)
  mkdirSync(outDir, { recursive: true })
  const created = []
  try {
    for (const [name, data] of outputs) {
      writeFileSync(join(outDir, name), data, { flag: 'wx' })
      created.push(name)
    }
  } catch (e) {
    for (const name of created) rmSync(join(outDir, name), { force: true })
    throw e
  }
}

// ---- the pinned browser --------------------------------------------------------------------------------------------------------------

// Runs in the blank page: decodes the two captures once, draws a picture for a layout, and checks a picture by decoding it again.
const PAGE_SOURCE = `(() => {
  const BG = ${JSON.stringify(BACKGROUND)}, INK = ${JSON.stringify(LABEL_COLOR)}, MAX = ${MAX_PICTURE_BYTES}
  const blobs = []
  const bytes = b64 => Uint8Array.from(atob(b64), c => c.charCodeAt(0))
  const paint = (ctx, lay, left, right, labels) => {
    ctx.fillStyle = 'rgb(' + BG.join(',') + ')'
    ctx.fillRect(0, 0, lay.width, lay.height)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(left, lay.left.x, lay.left.y, lay.left.w, lay.left.h)
    ctx.drawImage(right, lay.right.x, lay.right.y, lay.right.w, lay.right.h)
    if (labels) {
      ctx.fillStyle = 'rgb(' + INK.join(',') + ')'
      ctx.font = lay.labelSize + 'px sans-serif'
      ctx.textBaseline = 'middle'
      for (const l of lay.labels) ctx.fillText(l.text, l.x, l.y)
    }
  }
  const canvasFor = lay => { const c = document.createElement('canvas'); c.width = lay.width; c.height = lay.height; return c }
  const blockMean = (ctx, p) => {
    const d = ctx.getImageData(Math.max(0, p.x - 1), Math.max(0, p.y - 1), 3, 3).data
    const sum = [0, 0, 0]
    for (let i = 0; i < d.length; i += 4) { sum[0] += d[i]; sum[1] += d[i + 1]; sum[2] += d[i + 2] }
    return sum.map(v => v / (d.length / 4))
  }
  window.__load = (leftB64, rightB64) => { blobs.length = 0; blobs.push(new Blob([bytes(leftB64)], { type: 'image/png' }), new Blob([bytes(rightB64)], { type: 'image/png' })); return true }
  window.__encode = async (lay, mime, quality) => {
    const [left, right] = await Promise.all(blobs.map(b => createImageBitmap(b)))
    const canvas = canvasFor(lay)
    paint(canvas.getContext('2d', { alpha: false }), lay, left, right, true)
    const url = canvas.toDataURL(mime, quality)
    const b64 = url.slice(url.indexOf(',') + 1)
    const size = Math.floor(b64.length * 3 / 4) - (b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0)
    return size > MAX ? null : b64
  }
  window.__verify = async (b64, mime, lay, points) => {
    const decoded = await createImageBitmap(new Blob([bytes(b64)], { type: mime }))
    const [left, right] = await Promise.all(blobs.map(b => createImageBitmap(b)))
    const got = canvasFor(lay), want = canvasFor(lay)
    const gctx = got.getContext('2d', { alpha: false }), wctx = want.getContext('2d', { alpha: false })
    gctx.drawImage(decoded, 0, 0)
    paint(wctx, lay, left, right, false)
    let worst = 0
    for (const p of points) {
      const a = blockMean(gctx, p), b = blockMean(wctx, p)
      for (let i = 0; i < 3; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]))
    }
    return { width: decoded.width, height: decoded.height, worst }
  }
  return true
})()`

async function launch(chromium, switches) {
  const profile = mkdtempSync(join(tmpdir(), 'dc-offer-'))
  const args = [...switches, '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run',
    ...(process.getuid && process.getuid() === 0 ? ['--no-sandbox'] : []), 'about:blank']
  const child = spawn(chromium, args, { stdio: ['ignore', 'ignore', 'pipe'] })
  try {
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
        rmSync(profile, { force: true, maxRetries: 3, recursive: true })
      },
    }
  } catch (e) {
    child.kill('SIGKILL')
    rmSync(profile, { force: true, maxRetries: 3, recursive: true })
    throw e
  }
}

// One blank page with the network off: only data:, blob: and about: addresses may be requested.
async function blankPage(browser) {
  const { targetId } = await browser.send('Target.createTarget', { url: 'about:blank' })
  const { sessionId } = await browser.send('Target.attachToTarget', { targetId, flatten: true })
  const send = (method, params) => browser.send(method, params, sessionId)
  const blocked = []
  browser.on(m => {
    if (m.sessionId !== sessionId || m.method !== 'Fetch.requestPaused') return
    const url = m.params.request.url
    if (/^(data|blob|about):/.test(url)) send('Fetch.continueRequest', { requestId: m.params.requestId }).catch(() => {})
    else { blocked.push(url); send('Fetch.failRequest', { requestId: m.params.requestId, errorReason: 'BlockedByClient' }).catch(() => {}) }
  })
  await send('Fetch.enable', { patterns: [{ urlPattern: '*' }] })
  const evaluate = async expression => {
    const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
    if (r.exceptionDetails) throw new Error((r.exceptionDetails.exception || {}).description || r.exceptionDetails.text)
    return r.result.value
  }
  await evaluate(PAGE_SOURCE)
  return { evaluate, blocked }
}

// Draws and re-checks the picture of one option. Refuses (throws) on any failure; returns { data, suffix }.
export async function drawPicture(page, left, right) {
  await page.evaluate(`window.__load(${JSON.stringify(left.data.toString('base64'))}, ${JSON.stringify(right.data.toString('base64'))})`)
  const chosen = await chooseEncoding(async step => {
    const b64 = await page.evaluate(`window.__encode(${JSON.stringify(layout(left, right, step.scale))}, ${JSON.stringify(step.mime)}, ${step.quality ?? 'undefined'})`)
    return b64 === null ? null : Buffer.from(b64, 'base64')
  }, MAX_PICTURE_BYTES, async (data, step) => {
    const lay = layout(left, right, step.scale)
    if (pictureKind(data) !== step.mime) return 'the browser wrote a picture of the wrong type'
    const points = [...samplePoints(lay.left), ...samplePoints(lay.right)]
    const check = await page.evaluate(`window.__verify(${JSON.stringify(data.toString('base64'))}, ${JSON.stringify(step.mime)}, ${JSON.stringify(lay)}, ${JSON.stringify(points)})`)
    if (check.width !== lay.width || check.height !== lay.height) return 'the picture does not decode to the expected size'
    if (check.worst > toleranceFor(step.mime)) return 'the picture does not match its captures'
    return null
  })
  return chosen
}

// ---- the whole composition ------------------------------------------------------------------------------------------------------------

async function loadKit(kit) {
  let mod
  try { mod = await import(pathToFileURL(join(resolve(kit), 'capture.mjs')).href) } catch { throw new OfferError('the kit folder has no readable capture.mjs') }
  if (!mod.CHROMIUM_PIN || !Array.isArray(mod.CHROMIUM_SWITCHES) || typeof mod.verifyChromium !== 'function') {
    throw new OfferError('the kit folder has a capture.mjs without the Chromium pin and its checks')
  }
  return mod
}

// Everything is checked before anything is written; nothing is overwritten and nothing is written on a refusal.
export async function compose({ kit, chromium, out, options, pin }) {
  if (!kit || !chromium || !out) throw new OfferError('usage: compose_offer.mjs --kit KIT --chromium BINARY --out DIR --option PAGE CAPTURES (two or three times)')
  const numbered = checkOptions(options)
  const entries = []
  const drawn = []
  for (const { page, folder, n } of numbered) {
    if (!isRegularFile(page)) throw new OfferError(`${basename(page)}: the page file is missing`)
    if (!existsSync(folder) || !lstatSync(folder).isDirectory()) throw new OfferError(`${basename(page)}: the capture folder is missing`)
    const [left, right] = WIDTHS.map(w => readCapture(folder, w))
    drawn.push({ n, left, right })
    entries.push({ n, page: { file: basename(page), sha256: sha256(readFileSync(page)) }, left, right })
  }
  const shared = sharedKit(entries.flatMap(e => [e.left, e.right]))
  const taken = [MANIFEST, ...numbered.flatMap(({ n }) => ['.png', '.jpg'].map(s => `option-${n}-1280-375${s}`))].filter(f => existsSync(join(out, f)))
  if (taken.length) throw new OfferError(`refusing to overwrite ${taken.join(', ')}`)

  const kitModule = await loadKit(kit)
  let version
  try { version = kitModule.verifyChromium(chromium, pin || kitModule.CHROMIUM_PIN) } catch (e) { throw new OfferError(e.message) }

  const browser = await launch(chromium, kitModule.CHROMIUM_SWITCHES)
  const outputs = []
  try {
    const page = await blankPage(browser)
    for (const { n, left, right } of drawn) {
      const { data, suffix } = await drawPicture(page, left, right)
      const name = `option-${n}-1280-375${suffix}`
      outputs.push([name, data])
      entries.find(e => e.n === n).picture = { file: name, sha256: sha256(data), bytes: data.length }
    }
    if (page.blocked.length) throw new OfferError('the browser page asked for an outside address; refused')
  } finally {
    await browser.close()
  }
  const manifest = buildManifest({
    ...shared, composerChromiumVersion: version,
    entries: entries.map(e => ({
      n: e.n, page: e.page,
      captures: [e.left, e.right].map(c => ({ width: c.width, file: c.file, sha256: c.sha256 })),
      picture: e.picture,
    })),
  })
  outputs.push([MANIFEST, Buffer.from(`${JSON.stringify(manifest, null, 2)}\n`)])
  writeOutputs(out, outputs)
  return manifest
}

export function parseArgv(argv) {
  const args = { options: [] }
  for (let i = 0; i < argv.length; i++) {
    const flag = argv[i]
    if (flag === '--option') {
      if (i + 2 >= argv.length) throw new OfferError('--option needs a page file and a capture folder')
      args.options.push([argv[i + 1], argv[i + 2]])
      i += 2
    } else if (flag === '--kit' || flag === '--chromium' || flag === '--out') {
      if (i + 1 >= argv.length) throw new OfferError(`${flag} needs a value`)
      args[flag.slice(2)] = argv[++i]
    } else throw new OfferError(`unknown argument ${flag}`)
  }
  return args
}

async function main() {
  try {
    const manifest = await compose(parseArgv(process.argv.slice(2)))
    for (const e of manifest.options) console.log(`option ${e.n} ${e.picture.file} ${e.picture.bytes} bytes`)
    console.log(MANIFEST)
  } catch (e) {
    console.error(`refused: ${e.message}`)
    process.exit(1)
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) main()
