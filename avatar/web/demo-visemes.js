// Demo-only stand-in for Amazon Polly: guesses Polly-style viseme marks from plain text so
// the avatar can be tried without a backend. It is silent and only roughly right.
// In the app, use the real speech marks Polly returns (see README).

const DIGRAPHS = [
  ['tch', 'S'], ['sh', 'S'], ['ch', 'S'], ['th', 'T'], ['ph', 'f'], ['wh', 'u'], ['ck', 'k'], ['ng', 'k'], ['qu', 'k'],
  ['ee', 'i'], ['ea', 'i'], ['oo', 'u'], ['ou', 'a'], ['ow', 'o'], ['oa', 'o'], ['ai', 'e'], ['ay', 'e'], ['oi', 'O'], ['oy', 'O'],
  ['au', 'O'], ['aw', 'O'], ['ie', 'i'], ['igh', 'a'], ['er', '@'], ['ur', '@'], ['ir', '@'], ['ar', 'a'], ['or', 'O'],
];
const SINGLE = {
  a: 'E', e: 'E', i: 'i', o: 'O', u: '@', y: 'i',
  b: 'p', p: 'p', m: 'p', f: 'f', v: 'f', w: 'u', r: 'r',
  t: 't', d: 't', n: 't', l: 't', s: 's', z: 's', c: 'k', k: 'k', g: 'k', q: 'k', x: 's', j: 'S', h: '@',
};
const VOWELS = new Set(['a', 'e', 'E', 'i', 'o', 'O', 'u', '@']);

/** Returns Polly-shaped marks: [{time, type: 'viseme', value}] plus word marks. */
export function guessVisemes(text, { wordsPerMinute = 165 } = {}) {
  const marks = [];
  const scale = 165 / wordsPerMinute;
  let time = 120;
  const tokens = text.toLowerCase().match(/[a-z']+|[.,;:!?]/g) || [];
  for (const token of tokens) {
    if (/^[.,;:!?]$/.test(token)) {
      marks.push({ time: Math.round(time), type: 'viseme', value: 'sil' });
      time += (token === ',' ? 220 : 420) * scale;
      continue;
    }
    marks.push({ time: Math.round(time), type: 'word', value: token });
    let word = token.replace(/'/g, '');
    if (word.length > 3 && word.endsWith('e')) word = word.slice(0, -1);      // silent e
    let i = 0;
    let previous = null;
    while (i < word.length) {
      let viseme = null;
      for (const [letters, value] of DIGRAPHS) {
        if (word.startsWith(letters, i)) { viseme = value; i += letters.length; break; }
      }
      if (!viseme) { viseme = SINGLE[word[i]] || null; i += 1; }
      if (!viseme || viseme === previous) continue;
      marks.push({ time: Math.round(time), type: 'viseme', value: viseme });
      time += (VOWELS.has(viseme) ? 105 : 62) * scale;
      previous = viseme;
    }
    time += 55 * scale;
  }
  marks.push({ time: Math.round(time), type: 'viseme', value: 'sil' });
  return marks;
}
