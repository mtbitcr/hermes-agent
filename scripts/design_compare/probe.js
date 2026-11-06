(() => {
  const LANDMARKS = new Set(['header', 'nav', 'main', 'footer', 'aside', 'section', 'form', 'dialog'])
  const SIGNIFICANT = new Set([...LANDMARKS, 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'button', 'img', 'svg',
    'input', 'select', 'textarea', 'li', 'canvas', 'video'])
  const SKIP = new Set(['script', 'style', 'noscript', 'template'])
  const SIDES = ['top', 'right', 'bottom', 'left']

  const normalize = (s, max = Infinity) =>
    Array.from(String(s).normalize('NFKC').replace(/\s+/g, ' ').trim().toLowerCase()).slice(0, max).join('')
  const ownText = el => Array.from(el.childNodes).filter(n => n.nodeType === 3).map(n => n.nodeValue).join('')
  const parentOf = n => n.parentElement || (n.getRootNode && n.getRootNode().host)
    || (n.ownerDocument && n.ownerDocument.defaultView && n.ownerDocument.defaultView.frameElement) || null
  const styleOf = el => el.ownerDocument.defaultView.getComputedStyle(el)
  const FRAMES = new Set(['iframe', 'frame'])
  const headingOf = el => {
    const h = el.querySelector('h1,h2,h3,h4,h5,h6')
    return h ? normalize(h.textContent, 40) : ''
  }

  function describe(el) {
    const chain = []
    for (let p = parentOf(el); p; p = parentOf(p)) {
      if (LANDMARKS.has(p.localName)) chain.unshift(p.localName + (headingOf(p) ? `[${headingOf(p)}]` : ''))
    }
    return {
      chain: chain.join(' > '),
      role: el.getAttribute('role') || el.localName,
      name: normalize(el.getAttribute('aria-label') || el.getAttribute('alt') || ownText(el), 60),
    }
  }

  function assignKeys(els) {
    const seen = new Map()
    return els.map(el => {
      const { chain, role, name } = describe(el)
      const base = `${chain}/${role}:"${name}"`
      seen.set(base, (seen.get(base) || 0) + 1)
      return `${base}#${seen.get(base)}`
    })
  }

  const canvas = typeof document === 'undefined' ? null : document.createElement('canvas')
  function rgba(value) {
    if (!value || value === 'none') return null
    if (value === 'transparent') return [0, 0, 0, 0]
    const m = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)$/.exec(value)
    if (m) {
      const a = m[4] === undefined ? 1 : m[4].endsWith('%') ? parseFloat(m[4]) / 100 : parseFloat(m[4])
      return [+m[1], +m[2], +m[3], Math.round(a * 1000) / 1000]
    }
    if (/^url\(/.test(value)) return null
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    ctx.clearRect(0, 0, 1, 1)
    ctx.fillStyle = '#000'
    ctx.fillStyle = value
    ctx.fillRect(0, 0, 1, 1)
    const d = ctx.getImageData(0, 0, 1, 1).data
    return [d[0], d[1], d[2], Math.round((d[3] / 255) * 1000) / 1000]
  }

  const unquote = f => f.split(',')[0].trim().replace(/^["']|["']$/g, '').toLowerCase()
  const round = n => Math.round(n * 100) / 100

  function hiddenByOpacity(el, style = styleOf, cache = new Map()) {
    if (!el) return false
    if (!cache.has(el)) cache.set(el, parseFloat(style(el).opacity) <= 0 || hiddenByOpacity(parentOf(el), style, cache))
    return cache.get(el)
  }

  // Every element of the top body, of each open shadow root and of each same-origin child frame, with the page position of its viewport.
  function collect(root = document, ctx = { win: window, x: scrollX, y: scrollY }, out = { items: [], roots: [] }) {
    out.roots.push({ root, win: ctx.win })
    for (const el of (root.body || root).querySelectorAll('*')) {
      out.items.push({ el, ctx })
      if (el.shadowRoot) collect(el.shadowRoot, ctx, out)
      if (!FRAMES.has(el.localName)) continue
      const doc = el.contentDocument
      if (!doc) throw new Error(`unsupported content: a cross-origin or unreadable <${el.localName}> frame cannot be measured`)
      const r = el.getBoundingClientRect()
      const cs = ctx.win.getComputedStyle(el)
      collect(doc, { win: doc.defaultView, x: ctx.x + r.left + el.clientLeft + parseFloat(cs.paddingLeft), y: ctx.y + r.top + el.clientTop + parseFloat(cs.paddingTop) }, out)
    }
    return out
  }

  // Author inline !important beats every stylesheet, so those elements get the forced values inline; shadow roots get the stylesheet.
  function prepare(forced, css) {
    const { items, roots } = collect()
    for (const { root, win } of roots) {
      if (root === win.document) continue
      const sheet = new win.CSSStyleSheet()
      sheet.replaceSync(css)
      root.adoptedStyleSheets = [...root.adoptedStyleSheets, sheet]
    }
    for (const { el } of items) {
      const inline = el.style
      if (inline && Array.from(inline).some(p => inline.getPropertyPriority(p))) {
        for (const [name, value] of Object.entries(forced)) inline.setProperty(name, value, 'important')
      }
    }
  }

  const fontsReady = () => Promise.all(collect().roots.map(r => r.win.document.fonts.ready)).then(() => true)

  function measure() {
    const opacity = new Map()
    const loadedBy = new Map()
    const loadedIn = doc => {
      if (!loadedBy.has(doc)) {
        const set = new Set()
        doc.fonts.forEach(f => { if (f.status === 'loaded') set.add(unquote(f.family)) })
        loadedBy.set(doc, set)
      }
      return loadedBy.get(doc)
    }
    const found = []
    const masks = []
    for (const { el, ctx } of collect().items) {
      const tag = el.localName
      const cs = ctx.win.getComputedStyle(el)
      if (cs.webkitFontSmoothing !== 'antialiased' || cs.textRendering.toLowerCase() !== 'geometricprecision') {
        throw new Error(`rendering normalisation is not effective for <${tag}>: -webkit-font-smoothing ${cs.webkitFontSmoothing}, text-rendering ${cs.textRendering}`)
      }
      const svg = el.closest('svg')
      if (SKIP.has(tag) || (svg && svg !== el) || (el.parentElement && el.parentElement.closest('canvas,video'))) continue
      const r = el.getBoundingClientRect()
      if (r.width < 1 || r.height < 1 || cs.visibility !== 'visible' || hiddenByOpacity(el, styleOf, opacity)) continue
      const box = { x: round(r.left + ctx.x), y: round(r.top + ctx.y), width: round(r.width), height: round(r.height) }
      if (tag === 'canvas' || tag === 'video') masks.push(box)
      const border = SIDES.map(s => {
        const k = `border${s[0].toUpperCase()}${s.slice(1)}`
        return parseFloat(cs[`${k}Width`]) > 0 && cs[`${k}Style`] !== 'none' ? rgba(cs[`${k}Color`]) : null
      })
      const bg = rgba(cs.backgroundColor)
      const text = normalize(ownText(el))
      const own = text || (bg && bg[3] > 0) || cs.backgroundImage !== 'none' || border.some(c => c && c[3] > 0)
      if (!SIGNIFICANT.has(tag) && !own) continue
      const family = unquote(cs.fontFamily)
      found.push({
        el,
        record: {
          tag, text, pinned: cs.position === 'fixed' || cs.position === 'sticky', box,
          type: {
            family, size: parseFloat(cs.fontSize), weight: Number(cs.fontWeight),
            line_height: cs.lineHeight === 'normal' ? 'normal' : parseFloat(cs.lineHeight), face_loaded: loadedIn(el.ownerDocument).has(family),
          },
          colour: {
            color: rgba(cs.color), background_color: bg,
            border_top_color: border[0], border_right_color: border[1], border_bottom_color: border[2], border_left_color: border[3],
            text_decoration_color: cs.textDecorationLine === 'none' ? null : rgba(cs.textDecorationColor),
            fill: tag === 'svg' ? rgba(cs.fill) : null, stroke: tag === 'svg' ? rgba(cs.stroke) : null,
          },
        },
      })
    }
    const keys = assignKeys(found.map(f => f.el))
    return {
      elements: found.map((f, i) => ({ key: keys[i], ...f.record })),
      masks,
      pageHeight: Math.max(document.documentElement.scrollHeight, document.body.scrollHeight),
    }
  }

  const animationsIn = () => collect().roots.flatMap(({ root }) =>
    root.getAnimations ? root.getAnimations() : Array.from(root.children).flatMap(c => c.getAnimations({ subtree: true })))

  function settle() {
    for (const a of animationsIn()) {
      const end = a.effect ? a.effect.getComputedTiming().endTime : Infinity
      if (!Number.isFinite(end)) { a.cancel(); continue }
      try { a.finish() } catch { a.cancel() }
    }
    const running = animationsIn().filter(a => a.playState === 'running' || a.pending)
    if (running.length) throw new Error(`${running.length} animation(s) still running after settle`)
  }
  const hidePinned = () => collect().items.forEach(({ el, ctx }) => {
    const p = ctx.win.getComputedStyle(el).position
    if (p === 'fixed' || p === 'sticky') el.style.setProperty('visibility', 'hidden', 'important')
  })

  return { normalize, describe, assignKeys, hiddenByOpacity, prepare, fontsReady, measure, settle, hidePinned }
})()
