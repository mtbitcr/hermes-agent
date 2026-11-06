#!/usr/bin/env node
import { execFileSync, spawn } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { createServer } from 'node:http'
import { tmpdir } from 'node:os'
import { basename, extname, join, resolve, sep } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

// The binary hash of Chromium 153.0.8010.0 is not recorded in the plan; until it is set here the kit refuses to start.
export const CHROMIUM_PIN = { version: '153.0.8010.0', sha256: '' }
export const VIEWPORTS = [[1280, 800], [375, 812]]
export const MAX_PAGE_HEIGHT = 8000
export const CHROMIUM_SWITCHES = ['--headless=new', '--disable-gpu', '--disable-lcd-text', '--font-render-hinting=none',
  '--force-color-profile=srgb', '--hide-scrollbars', '--disable-features=PaintHolding', '--lang=en-US']
const STYLE = '*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important;'
  + 'scroll-behavior:auto!important;-webkit-font-smoothing:antialiased!important;text-rendering:geometricPrecision!important}'
const INJECT = `(()=>{const s=document.createElement('style');s.textContent=${JSON.stringify(STYLE)};`
  + 'const put=()=>document.documentElement&&(document.documentElement.appendChild(s),true);'
  + 'if(!put())new MutationObserver((_,o)=>put()&&o.disconnect()).observe(document,{childList:true})})()'
const PROTOTYPE_PREFIX = '/__prototype__/'
const TYPES = { '.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg', '.webp': 'image/webp', '.woff2': 'font/woff2', '.woff': 'font/woff' }
const dir = fileURLToPath(new URL('.', import.meta.url))
const sleep = ms => new Promise(r => setTimeout(r, ms))
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex')
const versionOf = text => (/\d+(\.\d+){3}/.exec(text) || [''])[0]

export function classifyRequest(url, origin) {
  try {
    const u = new URL(url)
    if (['data:', 'blob:', 'about:'].includes(u.protocol)) return 'allowed'
    return u.protocol === 'http:' && u.origin === origin ? 'allowed' : 'blocked'
  } catch {
    return 'blocked'
  }
}

export function assertSelfContained(blocked) {
  if (blocked.length) throw new Error(`the chosen design file is not self-contained (it requested ${blocked[0]})`)
}

export function verifyChromium(binary, pin, deps = {}) {
  if (!/^[0-9a-f]{64}$/.test(pin.sha256 || '')) throw new Error('Chromium binary hash is not pinned (missing hash): refusing to start')
  let bytes
  try { bytes = readFileSync(binary) } catch { throw new Error(`Chromium binary is not readable: ${binary}`) }
  if (sha256(bytes) !== pin.sha256) throw new Error('Chromium binary SHA-256 does not match the pin: refusing to start')
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

export const pageSlug = route => route.replace(/[^a-z0-9]+/gi, '-').replace(/^-|-$/g, '') || 'index'

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

async function capturePage(browser, stitcher, probeSource, { url, origin, width, height }) {
  const t = await openTarget(browser)
  const blocked = []
  const open = new Set()
  let lastActivity = Date.now()
  let onLoad
  const loaded = new Promise(r => { onLoad = r })
  browser.on(m => {
    if (m.sessionId !== t.sessionId) return
    const { requestId, request } = m.params
    lastActivity = Date.now()
    if (m.method === 'Fetch.requestPaused') {
      if (classifyRequest(request.url, origin) === 'blocked') {
        blocked.push(request.url)
        t.send('Fetch.failRequest', { requestId, errorReason: 'BlockedByClient' }).catch(() => {})
      } else t.send('Fetch.continueRequest', { requestId }).catch(() => {})
    } else if (m.method === 'Network.requestWillBeSent') open.add(requestId)
    else if (m.method === 'Network.loadingFinished' || m.method === 'Network.loadingFailed') open.delete(requestId)
    else if (m.method === 'Page.loadEventFired') onLoad()
  })
  try {
    for (const m of ['Page.enable', 'Network.enable']) await t.send(m)
    await t.send('Fetch.enable', { patterns: [{ urlPattern: '*' }] })
    await t.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false })
    await t.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-reduced-motion', value: 'reduce' }] })
    await t.send('Emulation.setTimezoneOverride', { timezoneId: 'UTC' })
    await t.send('Emulation.setLocaleOverride', { locale: 'en-US' })
    await t.send('Page.addScriptToEvaluateOnNewDocument', { source: INJECT })
    const nav = await t.send('Page.navigate', { url })
    if (nav.errorText) throw new Error(`cannot load ${url}: ${nav.errorText}`)
    await Promise.race([loaded, sleep(30000).then(() => { throw new Error(`timed out loading ${url}`) })])
    const { frameTree } = await t.send('Page.getFrameTree')
    const world = await t.send('Page.createIsolatedWorld', { frameId: frameTree.frame.id, worldName: 'design_compare' })
    const run = expr => t.evaluate(expr, world.executionContextId)
    await run('document.fonts.ready.then(() => true)')
    const deadline = Date.now() + 30000
    while (open.size || Date.now() - lastActivity < 500) {
      if (Date.now() > deadline) throw new Error(`network never went quiet for ${url}`)
      await sleep(50)
    }
    await run(frames)
    const probe = `(${probeSource}\n)`
    await run(`${probe}.settle()`)
    const measurement = await run(`${probe}.measure()`)
    const pageHeight = Math.min(measurement.pageHeight, MAX_PAGE_HEIGHT)
    const tiles = []
    for (let y = 0, n = 0; y < pageHeight; y += height, n++) {
      const top = Math.min(y, Math.max(pageHeight - height, 0))
      await run(`window.scrollTo(0, ${top}); ${frames}`)
      if (n === 1) await run(`${probe}.hidePinned()`)
      tiles.push({ top, data: (await t.send('Page.captureScreenshot', { format: 'png' })).data })
    }
    const png = await stitcher.evaluate(`(async (tiles, w, h) => {
      const c = document.createElement('canvas'); c.width = w; c.height = h
      const ctx = c.getContext('2d')
      for (const t of tiles) {
        const img = new Image(); img.src = 'data:image/png;base64,' + t.data; await img.decode()
        ctx.drawImage(img, 0, t.top)
      }
      return c.toDataURL('image/png').split(',')[1]
    })(${JSON.stringify(tiles)}, ${width}, ${pageHeight})`)
    return { blocked, measurement, pageHeight, png: Buffer.from(png, 'base64') }
  } finally {
    await t.close().catch(() => {})
  }
}

export async function capture({ side, siteDir, prototypeFile, pages, outDir, runId, candidateDigest, chromium, pin = CHROMIUM_PIN }) {
  if (side !== 'candidate' && side !== 'prototype') throw new Error('side must be candidate or prototype')
  const chromiumVersion = verifyChromium(chromium, pin)
  const probeSource = readFileSync(join(dir, 'probe.js'), 'utf8')
  const kitSha256 = sha256(Buffer.concat([readFileSync(join(dir, 'capture.mjs')), Buffer.from(probeSource)]))
  mkdirSync(outDir, { recursive: true })
  const server = await startServer({ siteDir, prototypeFile })
  const browser = await launch(chromium)
  const bundles = []
  try {
    const stitcher = await openTarget(browser)
    for (const route of pages) {
      const url = side === 'prototype' ? `${server.origin}${PROTOTYPE_PREFIX}${encodeURIComponent(basename(prototypeFile))}` : `${server.origin}${route}`
      for (const [width, height] of VIEWPORTS) {
        const shot = await capturePage(browser, stitcher, probeSource, { url, origin: server.origin, width, height })
        if (side === 'prototype') assertSelfContained(shot.blocked)
        const name = `${pageSlug(route)}-${width}`
        const bundle = buildBundle({
          kitSha256, chromiumVersion, runId, candidateDigest, page: route, width, pageHeight: shot.pageHeight, image: `${name}.png`,
          blockedRequests: shot.blocked, measurement: shot.measurement,
        })
        writeFileSync(join(outDir, `${name}.png`), shot.png)
        writeFileSync(join(outDir, `${name}.json`), `${JSON.stringify(bundle, null, 2)}\n`)
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
  main().catch(e => { console.error(e.message); process.exit(1) })
}
