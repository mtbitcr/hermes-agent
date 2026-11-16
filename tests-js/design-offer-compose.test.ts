import { spawnSync } from 'node:child_process'
import { createHash, randomBytes } from 'node:crypto'
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { deflateSync } from 'node:zlib'

import { afterAll, describe, expect, it } from 'vitest'

const skillScripts = new URL('../optional-skills/creative/design-offer/scripts/', import.meta.url)
const composeFile = fileURLToPath(new URL('compose_offer.mjs', skillScripts))
const kitFolder = fileURLToPath(new URL('../scripts/design_compare/', import.meta.url))
const compose = await import(/* @vite-ignore */ new URL('compose_offer.mjs', skillScripts).href)

const scratch = mkdtempSync(join(tmpdir(), 'design-offer-test-'))
afterAll(() => rmSync(scratch, { force: true, recursive: true }))

const KIT = 'a'.repeat(64)
const CHROMIUM = '153.0.8010.0'
const FOUR_MIB = 4 * 1024 * 1024
const sha = (data: Buffer | string) => createHash('sha256').update(data).digest('hex')

let counter = 0

const fresh = (name: string) => {
  const dir = join(scratch, `${name}-${++counter}`)

  mkdirSync(dir, { recursive: true })

  return dir
}

const crcTable = Array.from({ length: 256 }, (_, n) => {
  let c = n

  for (let k = 0; k < 8; k++) {
    c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  }

  return c >>> 0
})

const crc32 = (buf: Buffer) => {
  let c = 0xffffffff

  for (const b of buf) {
    c = crcTable[(c ^ b) & 0xff]! ^ (c >>> 8)
  }

  return (c ^ 0xffffffff) >>> 0
}

const chunk = (type: string, data: Buffer) => {
  const head = Buffer.alloc(8)

  head.writeUInt32BE(data.length, 0)
  head.write(type, 4, 'latin1')

  const tail = Buffer.alloc(4)

  tail.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), data])), 0)

  return Buffer.concat([head, data, tail])
}

// A valid RGB PNG: solid colour, or incompressible noise.
function makePng(width: number, height: number, fill: [number, number, number] | 'noise') {
  const ihdr = Buffer.alloc(13)

  ihdr.writeUInt32BE(width, 0)
  ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8
  ihdr[9] = 2

  const rows = Buffer.alloc((width * 3 + 1) * height)

  for (let y = 0; y < height; y++) {
    const at = y * (width * 3 + 1)

    if (fill === 'noise') {
      randomBytes(width * 3).copy(rows, at + 1)
    } else {
      for (let x = 0; x < width; x++) {
        rows.set(fill, at + 1 + x * 3)
      }
    }
  }

  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(rows)),
    chunk('IEND', Buffer.alloc(0)),
  ])
}

interface CaptureOpts {
  blocked?: string[]
  chromium?: string
  kit?: string
  leftHeight?: number
  noisy?: boolean
  rightHeight?: number
}

// A capture folder as the kit writes it for one page at both widths.
function fakeCapture(folder: string, o: CaptureOpts = {}) {
  mkdirSync(folder, { recursive: true })

  const sizes: [number, number, [number, number, number]][] = [
    [1280, o.leftHeight ?? 900, [200, 30, 30]],
    [375, o.rightHeight ?? 1400, [30, 30, 200]],
  ]

  for (const [width, height, color] of sizes) {
    writeFileSync(join(folder, `index-0a1b2c3d-${width}.png`), makePng(width, height, o.noisy ? 'noise' : color))
    writeFileSync(
      join(folder, `index-0a1b2c3d-${width}.json`),
      JSON.stringify({
        blocked_requests: o.blocked ?? [],
        candidate_digest: 'none',
        chromium_version: o.chromium ?? CHROMIUM,
        elements: [],
        image: `index-0a1b2c3d-${width}.png`,
        kit_sha256: o.kit ?? KIT,
        masks: [],
        page: '/',
        page_height: height,
        run_id: 'design-offer',
        schema: 'design_compare_bundle_v1',
        width,
      })
    )
  }

  return folder
}

function fakePage(root: string, name: string) {
  const dir = join(root, 'pages')

  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, name), `<!doctype html><title>${name}</title><p>option page</p>`)

  return join(dir, name)
}

function fakeOptions(count: number, o: CaptureOpts = {}): [string, string][] {
  const root = fresh('opts')

  return Array.from({ length: count }, (_, i) => [fakePage(root, `option-${i + 1}.html`), fakeCapture(join(root, `cap-${i + 1}`), o)])
}

const refusal = async (promise: Promise<unknown>) => {
  try {
    await promise
  } catch (e) {
    expect(e).toBeInstanceOf(compose.OfferError)

    return (e as Error).message
  }

  throw new Error('expected a refusal')
}

const check = (fn: () => unknown) => {
  try {
    fn()
  } catch (e) {
    expect(e).toBeInstanceOf(compose.OfferError)

    return (e as Error).message
  }

  throw new Error('expected a refusal')
}

describe('IHDR reading', () => {
  it('reads width and height from bytes 16-23 and needs nothing after them', () => {
    const png = makePng(1280, 900, [1, 2, 3])

    expect(compose.pngSize(png)).toEqual({ height: 900, width: 1280 })
    expect(compose.pngSize(png.subarray(0, 24))).toEqual({ height: 900, width: 1280 })
    expect(png.readUInt32BE(16)).toBe(1280)
    expect(png.readUInt32BE(20)).toBe(900)
  })

  it('refuses what is not a PNG', () => {
    const png = makePng(10, 10, [1, 2, 3])
    const wrongSignature = Buffer.from(png)

    wrongSignature[1] = 0x51

    const wrongChunk = Buffer.from(png)

    wrongChunk.write('IDAT', 12, 'latin1')

    for (const bad of [Buffer.alloc(0), png.subarray(0, 23), wrongSignature, wrongChunk, Buffer.from('GIF89a'.repeat(10))]) {
      expect(check(() => compose.pngSize(bad))).toContain('not a PNG')
    }
  })

  it('tells PNG and JPEG by their first bytes', () => {
    expect(compose.pictureKind(makePng(2, 2, [0, 0, 0]))).toBe('image/png')
    expect(compose.pictureKind(Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0]))).toBe('image/jpeg')
    expect(compose.pictureKind(Buffer.from('<html>'))).toBeNull()
  })
})

describe('option names and numbering', () => {
  it.each([0, 1, 4])('refuses %i options', count => {
    const options = Array.from({ length: count }, (_, i) => [`option-${(i % 3) + 1}.html`, 'x'])

    expect(check(() => compose.checkOptions(options))).toContain('two or three options')
  })

  it.each(['option-4.html', 'option-0.html', 'option-01.html', 'Option-1.html', 'option-1.htm', 'page-1.html', 'option-1.html.bak'])(
    'refuses the page name %s',
    bad => {
      expect(check(() => compose.checkOptions([[`dir/${bad}`, 'a'], ['dir/option-2.html', 'b']]))).toContain('option-<n>.html')
    }
  )

  it('refuses gaps and repeats', () => {
    expect(check(() => compose.checkOptions([['option-1.html', 'a'], ['option-3.html', 'b']]))).toContain('numbered 1, 2 and 3')
    expect(check(() => compose.checkOptions([['option-1.html', 'a'], ['option-1.html', 'b']]))).toContain('numbered 1, 2 and 3')
    expect(check(() => compose.checkOptions([['option-2.html', 'a'], ['option-3.html', 'b'], ['option-3.html', 'c']]))).toContain('numbered')
  })

  it('accepts two or three options in any order and sorts them by number', () => {
    const sorted = compose.checkOptions([['d/option-3.html', 'c'], ['d/option-1.html', 'a'], ['d/option-2.html', 'b']])

    expect(sorted.map((o: any) => o.n)).toEqual([1, 2, 3])
    expect(sorted.map((o: any) => o.folder)).toEqual(['a', 'b', 'c'])
    expect(compose.checkOptions([['option-2.html', 'b'], ['option-1.html', 'a']]).map((o: any) => o.n)).toEqual([1, 2])
  })
})

describe('bundle checks', () => {
  const rewrite = (folder: string, width: number, change: (bundle: any) => void) => {
    const file = join(folder, `index-0a1b2c3d-${width}.json`)
    const bundle = JSON.parse(readFileSync(file, 'utf8'))

    change(bundle)
    writeFileSync(file, JSON.stringify(bundle))
  }

  it('reads a capture with its hash, size and kit values', () => {
    const folder = fakeCapture(fresh('cap'))
    const left = compose.readCapture(folder, 1280)

    expect(left).toMatchObject({ chromium_version: CHROMIUM, file: 'index-0a1b2c3d-1280.png', height: 900, kit_sha256: KIT, width: 1280 })
    expect(left.sha256).toBe(sha(readFileSync(join(folder, 'index-0a1b2c3d-1280.png'))))
    expect(compose.readCapture(folder, 375).height).toBe(1400)
  })

  it.each([
    ['two widths of one page that disagree', (f: string) => rewrite(f, 375, b => { b.width = 1280 }), 'another width'],
    ['a blocked request', (f: string) => rewrite(f, 1280, b => { b.blocked_requests = ['http://x.example/'] }), 'blocked a request'],
    ['a missing blocked_requests list', (f: string) => rewrite(f, 1280, b => { delete b.blocked_requests }), 'blocked a request'],
    ['another schema', (f: string) => rewrite(f, 1280, b => { b.schema = 'other' }), 'not a capture bundle'],
    ['no kit hash', (f: string) => rewrite(f, 1280, b => { delete b.kit_sha256 }), 'no kit_sha256'],
    ['a malformed kit hash', (f: string) => rewrite(f, 1280, b => { b.kit_sha256 = 'abc' }), 'malformed kit_sha256'],
    ['no Chromium version', (f: string) => rewrite(f, 375, b => { b.chromium_version = '' }), 'no chromium_version'],
    ['a picture outside its folder', (f: string) => rewrite(f, 1280, b => { b.image = '../index-0a1b2c3d-1280.png' }), 'outside its folder'],
    ['a picture of another width name', (f: string) => rewrite(f, 1280, b => { b.image = 'index-0a1b2c3d-375.png' }), 'outside its folder'],
    ['a wrong page height', (f: string) => rewrite(f, 1280, b => { b.page_height = 901 }), 'does not match its bundle'],
    ['a page height that is not a whole number', (f: string) => rewrite(f, 1280, b => { b.page_height = '900' }), 'does not match its bundle'],
    ['a missing picture', (f: string) => rmSync(join(f, 'index-0a1b2c3d-1280.png')), 'is missing'],
    ['a second bundle for a width', (f: string) => writeFileSync(join(f, 'other-1280.json'), '{}'), 'exactly one 1280 px bundle'],
    ['no bundle for a width', (f: string) => rmSync(join(f, 'index-0a1b2c3d-375.json')), 'exactly one 375 px bundle'],
    ['an unreadable bundle', (f: string) => writeFileSync(join(f, 'index-0a1b2c3d-1280.json'), '{not json'), 'cannot be read'],
    ['a picture that is not a PNG', (f: string) => writeFileSync(join(f, 'index-0a1b2c3d-1280.png'), Buffer.from('not a png at all, no')), 'is not a PNG'],
    [
      'a picture of another width',
      (f: string) => writeFileSync(join(f, 'index-0a1b2c3d-1280.png'), makePng(1000, 900, [1, 1, 1])),
      'does not match its bundle',
    ],
  ])('refuses %s', (_name, change, message) => {
    const folder = fakeCapture(fresh('cap'))

    change(folder)

    expect(check(() => [1280, 375].forEach(w => compose.readCapture(folder, w)))).toContain(message)
  })

  it('refuses a picture that is a symbolic link', () => {
    const folder = fakeCapture(fresh('cap'))
    const real = join(folder, 'real.png')

    writeFileSync(real, readFileSync(join(folder, 'index-0a1b2c3d-1280.png')))
    rmSync(join(folder, 'index-0a1b2c3d-1280.png'))
    symlinkSync(real, join(folder, 'index-0a1b2c3d-1280.png'))

    expect(check(() => compose.readCapture(folder, 1280))).toContain('is missing')
  })

  it('refuses bundles of different kit hashes or Chromium versions', () => {
    const a = compose.readCapture(fakeCapture(fresh('cap')), 1280)
    const b = compose.readCapture(fakeCapture(fresh('cap'), { kit: 'b'.repeat(64) }), 1280)
    const c = compose.readCapture(fakeCapture(fresh('cap'), { chromium: '154.0.1.0' }), 1280)

    expect(check(() => compose.sharedKit([a, b]))).toContain('different capture kit hashes')
    expect(check(() => compose.sharedKit([a, c]))).toContain('different Chromium versions')
    expect(compose.sharedKit([a, a])).toEqual({ chromium_version: CHROMIUM, kit_sha256: KIT })
  })
})

describe('layout', () => {
  const left = { height: 900, width: 1280 }
  const right = { height: 1400, width: 375 }

  it('puts 1280 px left, 375 px right, labels above, with 40 px margins and a 72 px header', () => {
    const l = compose.layout(left, right, 1)

    expect(l.left).toEqual({ h: 900, w: 1280, x: 40, y: 72 })
    expect(l.right).toEqual({ h: 1400, w: 375, x: 1360, y: 72 })
    expect(l.width).toBe(40 * 3 + 1280 + 375)
    expect(l.height).toBe(72 + 40 + 1400)
    expect(l.labelSize).toBe(32)
    expect(l.labels).toEqual([{ text: '1280 px', x: 40, y: 36 }, { text: '375 px', x: 1360, y: 36 }])
  })

  it('shrinks everything evenly, never below one pixel', () => {
    const l = compose.layout(left, right, 0.8)

    expect(l.left).toEqual({ h: 720, w: 1024, x: 32, y: 58 })
    expect(l.right).toEqual({ h: 1120, w: 300, x: 1088, y: 58 })
    expect(l.width).toBe(32 * 3 + 1024 + 300)
    expect(l.height).toBe(58 + 32 + 1120)
    expect(l.labelSize).toBe(26)
    expect(compose.layout({ height: 1, width: 1 }, { height: 1, width: 1 }, 0.2).left.w).toBeGreaterThanOrEqual(1)
  })

  it('samples a grid of points that all lie inside their pane', () => {
    const l = compose.layout(left, right, 1)

    for (const pane of [l.left, l.right]) {
      const points = compose.samplePoints(pane)

      expect(points).toHaveLength(64)
      expect(new Set(points.map((p: any) => `${p.x},${p.y}`)).size).toBe(64)

      for (const p of points) {
        expect(p.x).toBeGreaterThanOrEqual(pane.x)
        expect(p.x).toBeLessThan(pane.x + pane.w)
        expect(p.y).toBeGreaterThanOrEqual(pane.y)
        expect(p.y).toBeLessThan(pane.y + pane.h)
      }
    }
  })

  it('is exact for PNG and within 24 per channel for JPEG', () => {
    expect(compose.toleranceFor('image/png')).toBe(0)
    expect(compose.toleranceFor('image/jpeg')).toBe(24)
  })
})

describe('4 MiB encode ladder', () => {
  it('goes PNG, JPEG 0.85, 0.70, 0.55, 0.40, then shrinks by 0.8 down to 0.2', () => {
    const ladder = compose.encodingLadder()
    const scales = [...new Set(ladder.map((s: any) => s.scale))] as number[]

    expect(ladder).toHaveLength(scales.length * 5)
    expect(scales).toHaveLength(8)
    expect(scales[0]).toBe(1)
    expect(scales[1]).toBeCloseTo(0.8)
    expect(scales[7]).toBeCloseTo(0.8 ** 7)
    expect(scales[7]).toBeGreaterThanOrEqual(0.2)
    expect(0.8 ** 8).toBeLessThan(0.2)
    expect(ladder.slice(0, 5).map((s: any) => [s.mime, s.quality])).toEqual([
      ['image/png', undefined],
      ['image/jpeg', 0.85],
      ['image/jpeg', 0.7],
      ['image/jpeg', 0.55],
      ['image/jpeg', 0.4],
    ])
    expect(ladder[5].scale).toBeCloseTo(0.8)
    expect(ladder[5].mime).toBe('image/png')
  })

  const run = async (fits: (step: any) => boolean) => {
    const calls: any[] = []

    const result = await compose.chooseEncoding(async (step: any) => {
      calls.push(step)

      return fits(step) ? Buffer.alloc(10) : null
    })

    return { calls, result }
  }

  it('takes the PNG when it fits and stops there', async () => {
    const { calls, result } = await run(() => true)

    expect(calls).toHaveLength(1)
    expect(result.suffix).toBe('.png')
  })

  it('takes the first JPEG quality that fits at the same scale', async () => {
    const { calls, result } = await run(s => s.mime === 'image/jpeg' && s.quality <= 0.7)

    expect(calls.map((s: any) => s.quality)).toEqual([undefined, 0.85, 0.7])
    expect(result.suffix).toBe('.jpg')
    expect(result.step.scale).toBe(1)
  })

  it('shrinks only when nothing fits at the current scale, and tries PNG first again', async () => {
    const { calls, result } = await run(s => s.scale < 0.7)

    expect(calls.findIndex((s: any) => s.scale < 1)).toBe(5)
    expect(result.step.scale).toBeCloseTo(0.64)
    expect(result.step.mime).toBe('image/png')
    expect(calls).toHaveLength(11)
  })

  it('treats a picture over 4 MiB as not fitting', async () => {
    const sizes = [FOUR_MIB + 1, FOUR_MIB]
    const result = await compose.chooseEncoding(async () => Buffer.alloc(sizes.shift() ?? 1))

    expect(result.data).toHaveLength(FOUR_MIB)
    expect(result.step.mime).toBe('image/jpeg')
  })

  it('refuses when no scale and quality fits', async () => {
    const calls: any[] = []

    const message = await refusal(
      compose.chooseEncoding(async (s: any) => {
        calls.push(s)

        return null
      })
    )

    expect(message).toContain('4 MiB')
    expect(calls).toHaveLength(40)
  })
})

describe('manifest and output files', () => {
  it('keeps the shape of today and adds the composer and the verified Chromium version', () => {
    const manifest = compose.buildManifest({
      chromium_version: CHROMIUM,
      composerChromiumVersion: CHROMIUM,
      entries: [{ captures: [], n: 1, page: { file: 'option-1.html', sha256: 'x' }, picture: { bytes: 3, file: 'option-1-1280-375.png', sha256: 'y' } }],
      kit_sha256: KIT,
    })

    expect(Object.keys(manifest)).toEqual(['kit_sha256', 'chromium_version', 'composer', 'composer_chromium_version', 'options'])
    expect(manifest.composer).toBe('compose_offer.mjs')
    expect(manifest.composer_chromium_version).toBe(CHROMIUM)
    expect(Object.keys(manifest.options[0])).toEqual(['captures', 'n', 'page', 'picture'])
  })

  it('writes every file exclusively and overwrites nothing', () => {
    const out = fresh('out')

    compose.writeOutputs(out, [['a.png', Buffer.from('a')], ['design-offer-manifest.json', Buffer.from('{}')]])
    expect(readdirSync(out).sort()).toEqual(['a.png', 'design-offer-manifest.json'])

    expect(check(() => compose.writeOutputs(out, [['b.png', Buffer.from('b')], ['a.png', Buffer.from('changed')]]))).toContain('refusing to overwrite a.png')
    expect(readFileSync(join(out, 'a.png'), 'utf8')).toBe('a')
    expect(existsSync(join(out, 'b.png'))).toBe(false)
  })
})

describe('refusals write nothing', () => {
  const fakeKit = (verify: string) => {
    const kit = fresh('kit')

    writeFileSync(
      join(kit, 'capture.mjs'),
      `import { writeFileSync } from 'node:fs'
export const CHROMIUM_PIN = { package: 'x', sha256: '${'0'.repeat(64)}', version: '1' }
export const CHROMIUM_SWITCHES = ['--headless=new']
export function verifyChromium(binary, pin) { writeFileSync(process.env.VERIFY_MARKER ?? '/dev/null', 'called'); ${verify} }
`
    )

    return kit
  }

  it('refuses a browser the kit does not verify, before any picture or file is made', async () => {
    const out = join(fresh('result'), 'offer')

    const message = await refusal(
      compose.compose({ chromium: '/nowhere', kit: fakeKit("throw new Error('Chromium binary SHA-256 does not match the pin: refusing to start')"), options: fakeOptions(2), out })
    )

    expect(message).toContain('does not match the pin')
    expect(existsSync(out)).toBe(false)
  })

  it('refuses any binary but the pinned one with the real kit', async () => {
    const binary = join(fresh('bin'), 'chromium')
    const out = join(fresh('result'), 'offer')

    writeFileSync(binary, 'not the pinned browser')

    const message = await refusal(compose.compose({ chromium: binary, kit: kitFolder, options: fakeOptions(2), out }))

    expect(message).toContain('does not match the pin')
    expect(existsSync(out)).toBe(false)
  })

  it('refuses a kit folder without a capture.mjs', async () => {
    const message = await refusal(compose.compose({ chromium: '/nowhere', kit: fresh('empty'), options: fakeOptions(2), out: join(fresh('r'), 'o') }))

    expect(message).toContain('capture.mjs')
  })

  it('checks the options and bundles first: the browser check is never reached', async () => {
    const marker = join(fresh('marker'), 'called')
    const kit = fakeKit('return "1"')

    process.env.VERIFY_MARKER = marker

    try {
      const out = join(fresh('result'), 'offer')
      const blocked = fakeOptions(2, { blocked: ['http://x.example/'] })

      const refusedMessages = [
        await refusal(compose.compose({ chromium: 'c', kit, options: fakeOptions(1), out })),
        await refusal(compose.compose({ chromium: 'c', kit, options: blocked, out })),
        await refusal(compose.compose({ chromium: 'c', kit, options: [[join(fresh('p'), 'option-1.html'), blocked[0]![1]], blocked[1]!], out })),
      ]

      expect(refusedMessages[0]).toContain('two or three')
      expect(refusedMessages[1]).toContain('blocked a request')
      expect(refusedMessages[2]).toContain('page file is missing')
      expect(existsSync(marker)).toBe(false)
      expect(existsSync(out)).toBe(false)
    } finally {
      delete process.env.VERIFY_MARKER
    }
  })

  it('refuses before the browser starts when an earlier result is in the way', async () => {
    const out = fresh('result')

    writeFileSync(join(out, 'design-offer-manifest.json'), 'earlier')

    const message = await refusal(compose.compose({ chromium: 'c', kit: fakeKit('return "1"'), options: fakeOptions(2), out }))

    expect(message).toContain('refusing to overwrite design-offer-manifest.json')
    expect(readFileSync(join(out, 'design-offer-manifest.json'), 'utf8')).toBe('earlier')
  })

  it('refuses mixed kit hashes', async () => {
    const root = fresh('mixed')

    const options: [string, string][] = [
      [fakePage(root, 'option-1.html'), fakeCapture(join(root, 'c1'))],
      [fakePage(root, 'option-2.html'), fakeCapture(join(root, 'c2'), { kit: 'b'.repeat(64) })],
    ]

    expect(await refusal(compose.compose({ chromium: 'c', kit: fakeKit('return "1"'), options, out: join(fresh('r'), 'o') }))).toContain(
      'different capture kit hashes'
    )
  })
})

describe('command line', () => {
  const run = (args: string[]) => spawnSync(process.execPath, [composeFile, ...args], { encoding: 'utf8', timeout: 60000 })

  it('exits 1 with "refused: ..." and writes nothing on a refusal', () => {
    const out = join(fresh('result'), 'offer')
    const [page, folder] = fakeOptions(1)[0]!
    const done = run(['--kit', kitFolder, '--chromium', '/nowhere', '--out', out, '--option', page, folder])

    expect(done.status).toBe(1)
    expect(done.stderr).toMatch(/^refused: an offer needs two or three options/)
    expect(done.stdout).toBe('')
    expect(existsSync(out)).toBe(false)
  })

  it.each([[[]], [['--bogus']], [['--kit']], [['--option', 'only-one']]])('refuses the arguments %j', args => {
    const done = run(args as string[])

    expect(done.status).toBe(1)
    expect(done.stderr).toMatch(/^refused: /)
  })

  it('refuses a browser that is not the pinned one, naming no page text', () => {
    const binary = join(fresh('bin'), 'chromium')
    const out = join(fresh('result'), 'offer')

    writeFileSync(binary, 'not the pinned browser')

    const args = ['--kit', kitFolder, '--chromium', binary, '--out', out]

    for (const [page, folder] of fakeOptions(2)) {
      args.push('--option', page, folder)
    }

    const done = run(args)

    expect(done.status).toBe(1)
    expect(done.stderr).toContain('does not match the pin')
    expect(`${done.stdout}${done.stderr}`).not.toContain('option page')
    expect(existsSync(out)).toBe(false)
  })

  it('imports without running anything and uses only Node built-ins', () => {
    const source = readFileSync(composeFile, 'utf8')
    const imports = [...source.matchAll(/^import .* from '([^']+)'/gm)].map(m => m[1])

    expect(imports.length).toBeGreaterThan(0)
    expect(imports.every(i => i!.startsWith('node:'))).toBe(true)
    expect(source).not.toMatch(/PIL|Pillow|sharp|puppeteer|playwright/)
  })
})

describe.skipIf(process.env.DESIGN_COMPARE_REAL_CHROMIUM !== '1')('real Chromium (opt-in, pinned binary from CHROME_PATH, shipped pin)', () => {
  const chromium = process.env.CHROME_PATH as string

  const composeReal = (options: [string, string][], out: string) => {
    const args = ['--kit', kitFolder, '--chromium', chromium, '--out', out]

    for (const [page, folder] of options) {
      args.push('--option', page, folder)
    }

    return spawnSync(process.execPath, [composeFile, ...args], { encoding: 'utf8', timeout: 300000 })
  }

  it('composes two options into one picture each, at the layout size, with a manifest of real hashes', () => {
    const options = fakeOptions(2)
    const out = join(fresh('result'), 'offer')
    const done = composeReal(options, out)

    expect(done.status, done.stderr).toBe(0)

    const manifest = JSON.parse(readFileSync(join(out, 'design-offer-manifest.json'), 'utf8'))
    const expected = compose.layout({ height: 900, width: 1280 }, { height: 1400, width: 375 }, 1)

    expect(manifest.composer).toBe('compose_offer.mjs')
    expect(manifest.composer_chromium_version).toBe(manifest.chromium_version)
    expect(manifest.options).toHaveLength(2)

    for (const entry of manifest.options) {
      const picture = readFileSync(join(out, entry.picture.file))

      expect(entry.picture.sha256).toBe(sha(picture))
      expect(entry.picture.bytes).toBe(picture.length)
      expect(picture.length).toBeLessThanOrEqual(FOUR_MIB)
      expect(compose.pictureKind(picture)).toBe('image/png')
      expect(compose.pngSize(picture)).toEqual({ height: expected.height, width: expected.width })
      expect(entry.page.sha256).toBe(sha(readFileSync(options[entry.n - 1]![0])))
    }

    expect(done.stdout).not.toContain('option page')
  })

  it('falls back to JPEG for a page too noisy for a 4 MiB PNG', () => {
    const options = fakeOptions(2, { leftHeight: 2500, noisy: true, rightHeight: 2500 })
    const out = join(fresh('result'), 'offer')
    const done = composeReal(options, out)

    expect(done.status, done.stderr).toBe(0)

    for (const name of readdirSync(out).filter(f => f.startsWith('option-'))) {
      const picture = readFileSync(join(out, name))

      expect(picture.length).toBeLessThanOrEqual(FOUR_MIB)
      expect(compose.pictureKind(picture)).toBe(name.endsWith('.jpg') ? 'image/jpeg' : 'image/png')
    }

    expect(readdirSync(out).some(f => f.endsWith('.jpg'))).toBe(true)
  })

  it('never overwrites an earlier result', () => {
    const options = fakeOptions(2)
    const out = join(fresh('result'), 'offer')

    expect(composeReal(options, out).status).toBe(0)

    const again = composeReal(options, out)

    expect(again.status).toBe(1)
    expect(again.stderr).toContain('refusing to overwrite')
  })
})
