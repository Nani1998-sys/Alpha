# Maya – talking avatar for the chat app

A full-length 3D character that stands, breathes, blinks, keeps eye contact and lip-syncs to
Amazon Polly in the browser (three.js). Everything is generated from CC0 MakeHuman assets
with Blender 4.5 + MPFB, so the character can be rebuilt and changed from code.

| File | What it is |
|---|---|
| `out/maya_pro.glb` | **The character to use.** 6.1 MB, 38,146 triangles, 52-joint Mixamo skeleton, 8 s `idle` clip, 52 ARKit blendshapes |
| `out/maya.glb` | Earlier head-and-shoulders version (no skeleton), kept for comparison |
| `web/avatar-runtime.js` | `AvatarStage`: loads the .glb and keeps it alive (see below) |
| `web/polly-lipsync.js` | Polly viseme speech marks → ARKit mouth shapes, with co-articulation |
| `web/index.html` | Demo page: framing, mood and state buttons, text box |
| `web/demo-visemes.js` | Demo-only rough text → viseme guesser, used when no Polly endpoint answers |
| `server/polly_server.py` | `POST /api/tts` → `{audio, marks}` from Amazon Polly; also serves the demo |
| `make_character.py` | Builds both characters and checks the written file |
| `previews/` | Screenshots from the three.js runtime (`maya_pro_*`) and Blender renders (`maya_*`) |

## Try it

```bash
pip install boto3
AWS_REGION=us-east-1 python avatar/server/polly_server.py --port 8080   # needs polly:SynthesizeSpeech
# open http://localhost:8080/web/
```

Without AWS credentials the page still works: it says so and mouths the words silently with
estimated timing. Add `?framing=full`, `?bg=transparent` or `?tts=https://your-api/tts` to
the URL to try options.

## Put it in your app

```js
import { AvatarStage } from './avatar-runtime.js';

const stage = new AvatarStage(document.getElementById('avatar'), {
  framing: 'medium',          // 'full' | 'medium' | 'closeup' | 'face'
  background: 'transparent',  // or 'studio', or any CSS colour
});
await stage.load('/assets/maya_pro.glb');
stage.setInsets({ top: 64, bottom: 96 });        // keep her clear of your header / chat box

// conversation loop
stage.setState('listening');                     // while the user types or talks
stage.setState('thinking');                      // while your LLM works
const { audio, marks } = await (await fetch('/api/tts', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: reply }),
})).json();
await stage.speak({ audio: base64ToBlob(audio), visemes: marks });   // back to idle when done
stage.setMood('happy');                          // 'neutral' | 'happy' | 'concerned' | 'surprised' | 'serious'
```

`speak()` also takes a URL, an `ArrayBuffer` or an `<audio>` element. If you already call
Polly elsewhere, you only need its speech marks: request `OutputFormat="json"` and
`SpeechMarkTypes=["viseme","word"]` with the **same voice and engine** as the audio. The
default voice is `Kajal` (Indian English, neural); set `POLLY_VOICE` / `POLLY_ENGINE`.
Browsers block audio until the user has clicked or typed on the page once.

### What `AvatarStage` does every frame

- **Body:** plays the `idle` clip from the file: breathing, a slow weight shift, small arm
  drift, feet planted.
- **Eyes:** keeps eye contact with the camera, with glances away whose timing depends on the
  state (more while thinking, fewer while listening). Small fixation shifts keep the eyes alive.
- **Head and neck:** follow the eyes part of the way, at a slower speed. While speaking, the
  head nods on emphasis, and the brows lift with it.
- **Blinks:** every 2–6 s, with occasional double blinks. She also blinks when the gaze shifts.
- **Mouth:** Polly visemes are blended so each shape arrives on its sound. Lip closures on
  p/b/m are held firmly, and the resting smile eases off while she talks.
- **Mood:** expressions blend in over about half a second.
- **Materials:** skin gets sheen and warm shadow fill, eyes a wet clear-coat, and hair soft
  alpha-to-coverage edges. Soft shadows sit on an invisible floor.
- **Camera:** gentle drift and a little parallax with the pointer.

## How real can this get?

This is a good, believable 3D character: it reads as a person from across the room and as
a well-made character up close. It will not be mistaken for a human on camera.
Human-indistinguishable talking characters today are AI-generated *video*, not 3D meshes:
services such as Tavus, Anam, HeyGen LiveAvatar and Simli stream a talking person over
WebRTC from your text or audio. They are mostly head-and-shoulders, cost per minute, and add
about 0.5–1.5 s of latency.

A practical combination: this 3D avatar for full-length, always-on presence (cheap, instant,
works offline), and a video-avatar API for close-up "face to face" moments, if the budget
allows. Both are driven by the same text from your chat backend.

## Building and changing the character

```bash
blender -b -P make_character.py -- --kind maya_pro --out out/maya_pro.glb --texture-size 1024
blender -b -P make_character.py -- --kind maya     --out out/maya.glb     --texture-size 1024 --preview-dir previews
```

It needs the MPFB extension and these CC0 MakeHuman asset packs: `makehuman_system_assets`,
`skins01`, `hair01` and `faceunits01` (the ARKit targets). The build refuses any asset that
the pack metadata does not mark CC0. After writing, it checks the .glb and exits non-zero if
any of these fail:

- size and triangle budget (< 8 MB, < 60k)
- 52 ARKit keys on the body, eyes, brows, lashes, teeth and tongue
- alpha modes
- embedded PNG/JPEG only, no Draco/KTX2
- height 1.6–1.85 m, feet at the origin
- one skin and the idle clip

Everything about the look lives in the `PRESETS` entry:

- **Body and face:** `macro` and `targets` (MPFB shape targets)
- **Assets:** `skin`, `hair`, `eyes`, `eyebrows`, `eyelashes`, `clothes`
- **Colour:** `*_grade` (texture colour grading) and `lips`
- **Pose:** `rest_pose` (arms down, stance)
- **Motion:** `idle` (clip length)
- **Shading detail:** `skin_normal` (pore detail) and `morph_normals` (lip and lid shading
  while talking, stored sparse on the body only)

Add new characters as new entries; `maya` is left as it was.

### Assets in `maya_pro` (all CC0)

| Part | Asset | Pack |
|---|---|---|
| Skin | `cutoff3d_indian_female_skin`, graded warmer and less red, with a derived pore normal map | skins01 |
| Hair | `culturalibre_hair_01`, lifted from black to very dark brown, body-weighted | hair01 |
| Eyes | `high-poly` with `brown` material; the see-through cornea shell is removed | system |
| Brows, lashes | `eyebrow012`, `eyelashes01` | system |
| Teeth, tongue | `teeth_base`, `tongue01` | system |
| Clothing | `female_elegantsuit01` recoloured to a plain navy blouse + charcoal skirt; `shoes04` as black ankle boots | system |
| Skeleton | MPFB `mixamo` rig (works with Mixamo animations) | MPFB |
| Face units | 52 ARKit targets | faceunits01 |

The face uses the same `targets` as `maya` except: `mouth-angles-up` 0.28 (0.55 parted the
lips), `cheek-volume` 0, and `head-fat-decr` 0.12.

### Known limits

- Only an idle clip is included. Gestures (wave, point, nod) can be retargeted from Mixamo,
  since the skeleton uses Mixamo bone names, and played with three.js `AnimationMixer`.
- Hair and clothing are skinned to the body but have no physics.
- glTF has no skin subsurface scattering. The runtime fakes it with sheen and a warm fill,
  which is convincing at chat distances but not in extreme close-up.
