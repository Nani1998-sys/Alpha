// Amazon Polly viseme speech marks -> ARKit blendshape weights.
//
// Polly returns one JSON object per line when you ask for SpeechMarkTypes=["viseme"]:
//   {"time":125,"type":"viseme","value":"p"}
// `time` is the viseme's onset in milliseconds from the start of the audio stream.
// VisemeTrack turns that list into a smooth, co-articulated set of weights for any time.

// Mouth shape for each Polly viseme, as ARKit blendshape weights (0..1).
// Left/right pairs are written once and expanded by expandPairs().
const SHAPES = {
  sil: {},
  p: { mouthPress: 0.55, mouthRollLower: 0.25, mouthRollUpper: 0.25, mouthClose: 0.05 },      // p b m
  f: { mouthRollLower: 0.55, mouthUpperUp: 0.30, jawOpen: 0.06, mouthLowerDown: 0.05 },       // f v
  T: { jawOpen: 0.16, tongueOut: 0.30, mouthLowerDown: 0.20, mouthUpperUp: 0.10 },            // th
  t: { jawOpen: 0.14, mouthLowerDown: 0.25, mouthUpperUp: 0.15, mouthStretch: 0.10 },         // t d n l
  S: { jawOpen: 0.12, mouthFunnel: 0.50, mouthPucker: 0.25, mouthUpperUp: 0.10 },             // sh ch j zh
  s: { jawOpen: 0.05, mouthStretch: 0.35, mouthSmile: 0.15, mouthLowerDown: 0.22, mouthUpperUp: 0.12 }, // s z
  k: { jawOpen: 0.22, mouthLowerDown: 0.15, mouthStretch: 0.08 },                             // k g ng
  r: { jawOpen: 0.12, mouthFunnel: 0.35, mouthPucker: 0.35 },                                 // r
  i: { jawOpen: 0.10, mouthSmile: 0.30, mouthStretch: 0.35, mouthLowerDown: 0.15 },           // ee, ih
  u: { jawOpen: 0.10, mouthPucker: 0.70, mouthFunnel: 0.35 },                                 // oo
  '@': { jawOpen: 0.20, mouthLowerDown: 0.10 },                                               // schwa
  a: { jawOpen: 0.52, mouthLowerDown: 0.30, mouthUpperUp: 0.10 },                             // ah
  e: { jawOpen: 0.30, mouthStretch: 0.25, mouthLowerDown: 0.20, mouthSmile: 0.10 },           // ay
  E: { jawOpen: 0.36, mouthStretch: 0.20, mouthLowerDown: 0.22 },                             // eh, ae
  o: { jawOpen: 0.28, mouthFunnel: 0.55, mouthPucker: 0.30 },                                 // oh
  O: { jawOpen: 0.42, mouthFunnel: 0.50, mouthPucker: 0.10 },                                 // aw
};

const PAIRED = ['mouthPress', 'mouthUpperUp', 'mouthLowerDown', 'mouthStretch', 'mouthSmile'];

function expandPairs(shape) {
  const out = {};
  for (const [key, value] of Object.entries(shape)) {
    if (PAIRED.includes(key)) {
      out[key + 'Left'] = value;
      out[key + 'Right'] = value;
    } else {
      out[key] = value;
    }
  }
  return out;
}

export const POLLY_VISEME_TO_ARKIT = Object.fromEntries(
  Object.entries(SHAPES).map(([viseme, shape]) => [viseme, expandPairs(shape)]),
);

// Every blendshape the lip sync can drive (used to reset the mouth between utterances).
export const LIPSYNC_KEYS = [...new Set(Object.values(POLLY_VISEME_TO_ARKIT).flatMap(Object.keys))];

/**
 * Parse Polly speech marks. Accepts the raw JSON-lines text Polly returns, an array of
 * JSON strings, or an array of already-parsed objects. Returns [{time, value}] for the
 * viseme marks and [{time, value}] for the word marks, both sorted by time.
 */
export function parsePollySpeechMarks(input) {
  let rows = input;
  if (typeof input === 'string') {
    rows = input.split('\n').map((line) => line.trim()).filter(Boolean);
  }
  const visemes = [];
  const words = [];
  for (const row of rows || []) {
    const mark = typeof row === 'string' ? JSON.parse(row) : row;
    if (!mark || typeof mark.time !== 'number') continue;
    if (mark.type === 'viseme' || mark.type === undefined) visemes.push({ time: mark.time, value: mark.value });
    else if (mark.type === 'word') words.push({ time: mark.time, value: mark.value });
  }
  visemes.sort((a, b) => a.time - b.time);
  words.sort((a, b) => a.time - b.time);
  return { visemes, words };
}

const smoothstep = (x) => {
  const t = Math.min(1, Math.max(0, x));
  return t * t * (3 - 2 * t);
};

/**
 * A viseme timeline that can be sampled at any time.
 *
 * Polly marks the *onset* of each viseme, so the mouth has to be on its way before the
 * mark: each shape is blended in over the `blendMs` before its onset and reaches full
 * strength exactly on it. Lip closures (p, b, m, f, v) are held to full strength because
 * a missed closure is the most visible lip-sync error.
 */
export class VisemeTrack {
  constructor(visemes, { blendMs = 85, intensity = 1.0 } = {}) {
    this.marks = (visemes || []).filter((m) => m.value in POLLY_VISEME_TO_ARKIT);
    this.blendMs = blendMs;
    this.intensity = intensity;
    this.cursor = 0;
    const last = this.marks[this.marks.length - 1];
    this.duration = last ? last.time + 350 : 0;
  }

  /** Weights at `timeMs`, written into `out` (keys not active are left out). */
  sample(timeMs, out = {}) {
    const marks = this.marks;
    if (!marks.length) return out;
    if (this.cursor >= marks.length || marks[this.cursor].time > timeMs) this.cursor = 0;
    while (this.cursor + 1 < marks.length && marks[this.cursor + 1].time <= timeMs) this.cursor += 1;

    const current = marks[this.cursor];
    const next = marks[this.cursor + 1];
    const before = timeMs < current.time;
    const from = before ? 'sil' : current.value;
    const to = before ? current : next;

    let blend = 0;
    if (to) {
      const gap = before ? this.blendMs : Math.max(1, to.time - current.time);
      const window = Math.min(this.blendMs, gap * 0.75);
      blend = smoothstep((timeMs - (to.time - window)) / window);
    } else if (timeMs > current.time + 250) {
      blend = smoothstep((timeMs - current.time - 250) / 150);         // relax after the last mark
    }

    const a = POLLY_VISEME_TO_ARKIT[from];
    const b = to ? POLLY_VISEME_TO_ARKIT[to.value] : POLLY_VISEME_TO_ARKIT.sil;
    for (const key of Object.keys(a)) out[key] = (out[key] || 0) + a[key] * (1 - blend) * this.intensity;
    for (const key of Object.keys(b)) out[key] = (out[key] || 0) + b[key] * blend * this.intensity;
    return out;
  }
}
