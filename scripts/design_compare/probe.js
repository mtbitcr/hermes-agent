(() => {
  const LANDMARKS = new Set(['header', 'nav', 'main', 'footer', 'aside', 'section', 'form', 'dialog'])
  const SIGNIFICANT = new Set([...LANDMARKS, 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'button', 'img', 'svg',
    'input', 'select', 'textarea', 'li', 'canvas', 'video'])
  const SKIP = new Set(['script', 'style', 'noscript', 'template'])
  const SIDES = ['top', 'right', 'bottom', 'left']

  const normalize = (s, max = Infinity) =>
    Array.from(String(s).normalize('NFKC').replace(/\s+/g, ' ').trim().toLowerCase()).slice(0, max).join('')
  const ownText = el => Array.from(el.childNodes).filter(n => n.nodeType === 3).map(n => n.nodeValue).join('')
  const headingOf = el => {
    const h = el.querySelector('h1,h2,h3,h4,h5,h6')
    return h ? normalize(h.textContent, 40) : ''
  }

  function describe(el) {
    const chain = []
    for (let p = el.parentElement; p; p = p.parentElement) {
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

  function measure() {
    const loaded = new Set()
    document.fonts.forEach(f => { if (f.status === 'loaded') loaded.add(unquote(f.family)) })
    const found = []
    const masks = []
    for (const el of document.body.querySelectorAll('*')) {
      const tag = el.localName
      const svg = el.closest('svg')
      if (SKIP.has(tag) || (svg && svg !== el) || (el.parentElement && el.parentElement.closest('canvas,video'))) continue
      const r = el.getBoundingClientRect()
      const cs = getComputedStyle(el)
      if (r.width < 1 || r.height < 1 || cs.visibility !== 'visible' || parseFloat(cs.opacity) <= 0) continue
      const box = { x: round(r.left + scrollX), y: round(r.top + scrollY), width: round(r.width), height: round(r.height) }
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
            line_height: cs.lineHeight === 'normal' ? 'normal' : parseFloat(cs.lineHeight), face_loaded: loaded.has(family),
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

  const settle = () => document.getAnimations().forEach(a => { try { a.finish() } catch { /* infinite animation */ } })
  const hidePinned = () => document.querySelectorAll('body *').forEach(el => {
    const p = getComputedStyle(el).position
    if (p === 'fixed' || p === 'sticky') el.style.setProperty('visibility', 'hidden', 'important')
  })

  return { normalize, describe, assignKeys, measure, settle, hidePinned }
})()
