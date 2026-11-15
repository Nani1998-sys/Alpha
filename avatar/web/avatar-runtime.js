// AvatarStage - puts a make_character.py avatar (.glb) on screen and keeps it alive.
//
//   const stage = new AvatarStage(document.getElementById('stage'));
//   await stage.load('./maya_pro.glb');
//   await stage.speak({ audio: mp3Blob, visemes: pollySpeechMarks });
//
// What it does every frame, in this order:
//   1. plays the "idle" clip from the file (breathing, weight shift)
//   2. adds head and neck motion on top: eye contact, glances, nods while speaking
//   3. drives the ARKit blendshapes: blinks, eye direction, mood, lip sync
//   4. moves the camera to the chosen framing and renders with soft shadows
//
// Needs three.js r163 or newer (import map: "three" and "three/addons/").

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { VisemeTrack, parsePollySpeechMarks } from './polly-lipsync.js';

const DEG = Math.PI / 180;
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const damp = (current, target, lambda, dt) => current + (target - current) * (1 - Math.exp(-lambda * dt));
const rand = (lo, hi) => lo + Math.random() * (hi - lo);
// Slow organic wobble in -1..1 (sum of sines that never line up).
const drift = (t, seed) =>
  (Math.sin(t * 0.73 + seed * 1.7) + 0.6 * Math.sin(t * 1.31 + seed * 2.9) + 0.3 * Math.sin(t * 2.17 + seed * 4.3)) / 1.9;

// Resting face per mood, as ARKit weights. Left/right pairs are written once.
const MOODS = {
  neutral: { mouthSmile: 0.14, cheekSquint: 0.06, browInnerUp: 0.06 },
  happy: { mouthSmile: 0.42, cheekSquint: 0.28, eyeSquint: 0.12, browInnerUp: 0.05 },
  concerned: { browInnerUp: 0.45, mouthFrown: 0.10, eyeSquint: 0.06 },
  surprised: { browInnerUp: 0.45, browOuterUp: 0.45, eyeWide: 0.30, jawOpen: 0.06 },
  serious: { browDown: 0.12, mouthPress: 0.08 },
};

// What each conversation state adds: face offsets, gaze habits, head motion, blink rate.
const STATES = {
  idle: { face: {}, eyeContact: 0.7, away: [6, 14, 5], hold: [1.5, 4], awayHold: [0.5, 1.4], motion: 0.7, tilt: 0, blink: [2.4, 5.5] },
  listening: { face: { browInnerUp: 0.12, mouthSmile: 0.06 }, eyeContact: 0.92, away: [4, 8, 3], hold: [2.5, 5], awayHold: [0.3, 0.8], motion: 0.5, tilt: 3, blink: [2.2, 5] },
  thinking: { face: { browDown: 0.10, mouthPress: 0.12, eyeSquint: 0.08 }, eyeContact: 0.25, away: [10, 18, 9], hold: [0.6, 1.4], awayHold: [1.0, 2.4], motion: 0.6, tilt: -2, blink: [3, 7] },
  speaking: { face: { browInnerUp: 0.04 }, eyeContact: 0.78, away: [5, 11, 4], hold: [1.5, 3.5], awayHold: [0.3, 0.9], motion: 1.6, tilt: 0, blink: [1.8, 4.2] },
};

const PAIRED = ['mouthSmile', 'cheekSquint', 'eyeSquint', 'browOuterUp', 'browDown', 'mouthFrown', 'eyeWide', 'mouthPress'];

// Camera framings as fractions of the character's height: [bottom, top, width in metres].
const FRAMINGS = {
  full: [-0.04, 1.06, 0.95],
  medium: [0.50, 1.07, 0.75],
  closeup: [0.76, 1.05, 0.50],
  face: [0.85, 1.035, 0.34],
};

function addShape(out, shape, amount = 1) {
  for (const key in shape) {
    const value = shape[key] * amount;
    if (PAIRED.includes(key)) {
      out[key + 'Left'] = (out[key + 'Left'] || 0) + value;
      out[key + 'Right'] = (out[key + 'Right'] || 0) + value;
    } else {
      out[key] = (out[key] || 0) + value;
    }
  }
  return out;
}

export class AvatarStage {
  /**
   * @param {HTMLElement} container  element the canvas fills
   * @param {object} [options]
   * @param {'studio'|'transparent'|string} [options.background='studio']  'transparent' shows your page behind the avatar; any CSS colour also works
   * @param {'full'|'medium'|'closeup'|'face'} [options.framing='medium']
   * @param {boolean} [options.followPointer=true]  small camera parallax and gaze response to the pointer
   * @param {boolean} [options.shadows=true]
   * @param {number} [options.maxPixelRatio=2]
   * @param {(state: string) => void} [options.onState]
   */
  constructor(container, options = {}) {
    this.container = container;
    this.options = { background: 'studio', framing: 'medium', followPointer: true, shadows: true, maxPixelRatio: 2, ...options };
    this.state = 'idle';
    this.mood = 'neutral';
    this.orbit = this.options.orbit || 0;
    this.height = 1.7;
    this._time = 0;
    this._pointer = new THREE.Vector2();
    this._pointerSmooth = new THREE.Vector2();
    this._faceMeshes = [];
    this._keys = [];
    this._weights = {};
    this._expression = {};
    this._viseme = {};
    this._manual = null;
    this._speech = null;
    this._bones = {};
    this._rest = new Map();
    this._animated = new Set();
    this._blink = { timer: rand(1, 3), phase: -1, value: 0, again: false };
    this._gaze = { timer: 0, awayYaw: 0, awayPitch: 0, eyeYaw: 0, eyePitch: 0, headYaw: 0, headPitch: 0 };
    this._emphasis = { value: 0, cooldown: 0, lastJaw: 0 };
    this._cam = { position: new THREE.Vector3(0, 1.5, 3), target: new THREE.Vector3(0, 1.4, 0), ready: false };
    this._tmp = { q1: new THREE.Quaternion(), q2: new THREE.Quaternion(), q3: new THREE.Quaternion(), e: new THREE.Euler(), v1: new THREE.Vector3(), v2: new THREE.Vector3() };

    const transparent = this.options.background === 'transparent';
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: transparent, powerPreference: 'high-performance' });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.shadowMap.enabled = !!this.options.shadows;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.domElement.style.cssText = 'display:block;width:100%;height:100%';
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(24, 1, 0.05, 50);
    this._buildLights();
    this._buildBackdrop();

    this._onPointer = (event) => {
      const rect = this.container.getBoundingClientRect();
      this._pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -(((event.clientY - rect.top) / rect.height) * 2 - 1));
    };
    window.addEventListener('pointermove', this._onPointer, { passive: true });
    this._resizeObserver = new ResizeObserver(() => this.resize());
    this._resizeObserver.observe(container);
    this.resize();

    this._clock = new THREE.Clock();
    this.renderer.setAnimationLoop(() => this._frame(Math.min(this._clock.getDelta(), 0.1)));
  }

  // ---------------------------------------------------------------- public API

  /** Load a character .glb written by make_character.py. */
  async load(url) {
    const gltf = await new GLTFLoader().loadAsync(url);
    if (this.root) this.scene.remove(this.root);
    this.root = gltf.scene;
    this.scene.add(this.root);

    let top = 0;
    this._faceMeshes = [];
    this.root.traverse((node) => {
      if (node.isBone) this._registerBone(node);
      if (!node.isMesh) return;
      node.frustumCulled = false;                       // skinned + morphed bounds are unreliable
      this._upgradeMaterial(node);
      const position = node.geometry.attributes.position;
      for (let i = 1; i < position.count * 3; i += 3) top = Math.max(top, position.array[i]);
      if (node.morphTargetDictionary) this._faceMeshes.push(node);
    });
    this.height = top || 1.7;
    this._keys = [...new Set(this._faceMeshes.flatMap((mesh) => Object.keys(mesh.morphTargetDictionary)))];

    this.mixer = null;
    this._animated.clear();
    const clip = gltf.animations.find((a) => /idle/i.test(a.name)) || gltf.animations[0];
    if (clip) {
      this.mixer = new THREE.AnimationMixer(this.root);
      this.mixer.clipAction(clip).play();
      for (const track of clip.tracks) this._animated.add(track.name.split('.')[0]);
    }

    this._placeLights();
    this.setFraming(this.options.framing, true);
    return this;
  }

  /**
   * Speak with lip sync.
   * @param {object} line
   * @param {string|Blob|ArrayBuffer|HTMLAudioElement} [line.audio]  the Polly audio (URL, Blob, bytes or an <audio>). Leave out for a silent run.
   * @param {Array|string} line.visemes  Polly speech marks: raw JSON-lines text, or an array of marks
   * @param {number} [line.intensity=1]  mouth movement scale
   * @returns {Promise<void>} resolves when the line has finished (or was stopped)
   */
  speak({ audio, visemes, intensity = 1 } = {}) {
    this.stopSpeaking();
    const marks = Array.isArray(visemes) && visemes.length && visemes[0].type === undefined && typeof visemes[0] === 'object'
      ? { visemes }
      : parsePollySpeechMarks(visemes);
    const track = new VisemeTrack(marks.visemes, { intensity });
    const before = this.state;

    return new Promise((resolve) => {
      const speech = { track, element: null, url: null, started: performance.now(), finished: false };
      speech.finish = () => {
        if (speech.finished) return;
        speech.finished = true;
        if (speech.element) speech.element.pause();
        if (speech.url) URL.revokeObjectURL(speech.url);
        if (this._speech === speech) {
          this._speech = null;
          this.setState(before === 'speaking' ? 'idle' : before);
        }
        resolve();
      };

      if (audio) {
        let element = audio;
        if (!(audio instanceof HTMLMediaElement)) {
          let source = audio;
          if (audio instanceof ArrayBuffer || ArrayBuffer.isView(audio)) source = new Blob([audio], { type: 'audio/mpeg' });
          if (source instanceof Blob) source = speech.url = URL.createObjectURL(source);
          element = new Audio(source);
        }
        speech.element = element;
        speech.clock = () => element.currentTime * 1000;
        element.addEventListener('ended', speech.finish, { once: true });
        element.addEventListener('error', speech.finish, { once: true });
        element.play().catch((error) => {
          console.warn('AvatarStage: audio could not start (the browser needs a user gesture first)', error);
          speech.finish();
        });
      } else {
        speech.clock = () => performance.now() - speech.started;
        speech.silentEnd = track.duration;
      }
      this._speech = speech;
      this.setState('speaking');
    });
  }

  stopSpeaking() {
    if (this._speech) this._speech.finish();
  }

  /** 'idle' | 'listening' | 'thinking' | 'speaking' */
  setState(state) {
    if (!STATES[state] || state === this.state) return;
    this.state = state;
    this._gaze.timer = 0;
    if (this.options.onState) this.options.onState(state);
  }

  /** 'neutral' | 'happy' | 'concerned' | 'surprised' | 'serious' */
  setMood(mood) {
    if (MOODS[mood]) this.mood = mood;
  }

  /** 'full' | 'medium' | 'closeup' | 'face' */
  setFraming(name, immediate = false) {
    if (!FRAMINGS[name]) return;
    this.framing = name;
    if (immediate) this._cam.ready = false;
  }

  /**
   * Keep the character clear of your own UI: pixels covered at the top and bottom of the
   * container (toolbars, chat input). The framing is fitted into the space in between.
   */
  setInsets({ top = 0, bottom = 0 } = {}) {
    this._insets = { top, bottom };
  }

  /** Turn the camera around the character, in degrees (0 = front, 30 = three-quarter). */
  setOrbit(degrees) {
    this.orbit = degrees;
  }

  /** Override blendshapes by hand ({jawOpen: 1}); pass null to hand control back. */
  setBlendshapes(weights) {
    this._manual = weights;
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, this.options.maxPixelRatio));
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  dispose() {
    this.stopSpeaking();
    this.renderer.setAnimationLoop(null);
    this._resizeObserver.disconnect();
    window.removeEventListener('pointermove', this._onPointer);
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }

  // ---------------------------------------------------------------- scene

  _buildLights() {
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.5;
    pmrem.dispose();

    this.keyLight = new THREE.DirectionalLight(0xfff0de, 2.3);
    this.keyLight.castShadow = !!this.options.shadows;
    this.keyLight.shadow.mapSize.set(2048, 2048);
    this.keyLight.shadow.bias = -0.0004;
    this.keyLight.shadow.normalBias = 0.012;
    this.keyLight.shadow.radius = 5;
    this.fillLight = new THREE.DirectionalLight(0xdbe6ff, 0.55);
    this.rimLight = new THREE.DirectionalLight(0xffffff, 1.5);
    this.scene.add(this.keyLight, this.keyLight.target, this.fillLight, this.fillLight.target, this.rimLight, this.rimLight.target);
    this._placeLights();
  }

  _placeLights() {
    const h = this.height;
    const aim = new THREE.Vector3(0, h * 0.6, 0);
    this.keyLight.target.position.copy(aim);
    this.keyLight.position.set(1.3, h + 2.0, 2.8);
    this.fillLight.target.position.copy(aim);
    this.fillLight.position.set(-2.6, h * 0.8, 2.0);
    this.rimLight.target.position.copy(aim);
    this.rimLight.position.set(-1.4, h + 1.0, -2.6);
    const cam = this.keyLight.shadow.camera;
    cam.left = -0.9; cam.right = 0.9; cam.top = h * 0.55; cam.bottom = -h * 0.75; cam.near = 0.5; cam.far = 9;
    cam.updateProjectionMatrix();
  }

  _buildBackdrop() {
    const background = this.options.background;
    if (background === 'studio') {
      const canvas = document.createElement('canvas');
      canvas.width = 16; canvas.height = 256;
      const ctx = canvas.getContext('2d');
      const gradient = ctx.createLinearGradient(0, 0, 0, 256);
      gradient.addColorStop(0, '#c9d0d8');
      gradient.addColorStop(0.55, '#aab3bd');
      gradient.addColorStop(1, '#7f8892');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 16, 256);
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      this.scene.background = texture;
    } else if (background !== 'transparent') {
      this.scene.background = new THREE.Color(background);
    }
    // Floor that only shows the character's shadow, so it sits on any backdrop.
    const floor = new THREE.Mesh(new THREE.CircleGeometry(4, 48), new THREE.ShadowMaterial({ opacity: 0.24 }));
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
  }

  _registerBone(bone) {
    const short = bone.name.replace(/^mixamorig:?/i, '');
    this._bones[short] = bone;
    this._rest.set(bone, bone.quaternion.clone());
  }

  // GLTFLoader gives every material the same plain PBR setup. Skin, eyes, hair and cloth
  // each need their own treatment to stop looking like painted plastic.
  _upgradeMaterial(mesh) {
    const old = mesh.material;
    const kind = (old.name || mesh.name).replace(/[0-9]+$/, '');
    let material = old;
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    if (kind === 'skin') {
      material = new THREE.MeshPhysicalMaterial({
        map: old.map, normalMap: old.normalMap, metalness: 0,
        roughnessMap: old.roughnessMap, roughness: old.roughnessMap ? 1.0 : 0.52,   // the map makes the lips glossier
        specularIntensity: 0.45,
        sheen: 0.18, sheenColor: new THREE.Color(1.0, 0.72, 0.62), sheenRoughness: 0.5,     // soft light at grazing angles
        emissive: new THREE.Color(0.5, 0.2, 0.14), emissiveMap: old.map, emissiveIntensity: 0.035, // warm, never-black shadows (cheap stand-in for subsurface light)
      });
      material.color.setRGB(1.0, 0.965, 0.93);
      if (old.normalMap) material.normalScale.set(0.6, 0.6);
    } else if (kind === 'eyes') {
      material = new THREE.MeshPhysicalMaterial({
        map: old.map, roughness: 0.35, metalness: 0, clearcoat: 1, clearcoatRoughness: 0.04, envMapIntensity: 0.9,
      });
    } else if (kind === 'teeth') {
      old.roughness = 0.28;
      mesh.castShadow = false;
    } else if (kind === 'tongue') {
      old.roughness = 0.4;
      mesh.castShadow = false;
    } else if (kind === 'hair') {
      old.roughness = 0.62;
      old.envMapIntensity = 0.55;
      old.alphaToCoverage = false;                      // alpha-to-coverage drew dark contour lines in the strands
      old.side = THREE.DoubleSide;
      old.shadowSide = THREE.DoubleSide;
    } else if (kind === 'eyebrows' || kind === 'eyelashes') {
      old.depthWrite = false;
      old.roughness = 0.8;
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      mesh.renderOrder = 2;
    } else if (kind === 'clothes') {
      material = new THREE.MeshPhysicalMaterial({
        map: old.map, normalMap: old.normalMap, roughness: Math.max(old.roughness, 0.4), metalness: 0, side: old.side,
        sheen: old.roughness > 0.6 ? 0.5 : 0.0, sheenColor: new THREE.Color(0.55, 0.6, 0.75), sheenRoughness: 0.6,
      });
      material.shadowSide = THREE.DoubleSide;
    }
    if (material !== old) {
      material.name = old.name;
      mesh.material = material;
      old.dispose();
    }
  }

  // ---------------------------------------------------------------- per-frame

  _frame(dt) {
    this._time += dt;
    if (this.root) {
      if (this.mixer) this.mixer.update(dt);
      const jaw = this._updateSpeech(dt);
      this._updateGaze(dt);
      this._updateHead(dt, jaw);
      this._updateFace(dt);
      this._updateCamera(dt);
    }
    this.renderer.render(this.scene, this.camera);
  }

  // Lip sync: sample the viseme track at the audio clock and ease towards it.
  _updateSpeech(dt) {
    const target = {};
    const speech = this._speech;
    if (speech) {
      const now = speech.clock();
      speech.track.sample(now + 20, target);            // look ahead by roughly one frame
      if (speech.silentEnd !== undefined && now > speech.silentEnd) speech.finish();
    }
    const keys = new Set([...Object.keys(target), ...Object.keys(this._viseme)]);
    for (const key of keys) {
      const value = damp(this._viseme[key] || 0, target[key] || 0, 26, dt);
      if (value < 0.002 && !target[key]) delete this._viseme[key];
      else this._viseme[key] = value;
    }
    // A syllable that opens the jaw quickly after a pause reads as emphasis: nod and lift the brows.
    const jaw = target.jawOpen || 0;
    const emphasis = this._emphasis;
    emphasis.cooldown -= dt;
    emphasis.value = Math.max(0, emphasis.value - dt * 2.6);
    if (speech && jaw > 0.38 && emphasis.lastJaw < 0.2 && emphasis.cooldown <= 0 && Math.random() < 0.55) {
      emphasis.value = 1;
      emphasis.cooldown = rand(0.7, 1.6);
    }
    emphasis.lastJaw = damp(emphasis.lastJaw, jaw, 8, dt);
    return jaw;
  }

  // Where the eyes want to point (degrees; yaw + = character's left, pitch + = up).
  _updateGaze(dt) {
    const gaze = this._gaze;
    const style = STATES[this.state];
    gaze.timer -= dt;
    if (gaze.timer <= 0) {
      const away = Math.random() > style.eyeContact;
      if (away) {
        const [yawMin, yawMax, pitchMax] = style.away;
        gaze.awayYaw = rand(yawMin, yawMax) * (Math.random() < 0.5 ? -1 : 1);
        gaze.awayPitch = this.state === 'thinking' ? rand(2, pitchMax) : rand(-pitchMax, pitchMax * 0.6);
        gaze.timer = rand(...style.awayHold);
        if (Math.random() < 0.5) this._blink.timer = 0;   // people blink when they shift gaze
      } else {
        gaze.awayYaw = rand(-1.2, 1.2);                   // tiny fixation shifts keep the eyes alive
        gaze.awayPitch = rand(-0.8, 0.8);
        gaze.timer = rand(...style.hold);
      }
    }

    // Direction from the head to the camera = eye contact.
    let yaw = 0;
    let pitch = 0;
    const head = this._bones.Head;
    if (head) {
      const eyes = head.getWorldPosition(this._tmp.v1);
      eyes.y += this.height * 0.045;
      const to = this._tmp.v2.copy(this.camera.position).sub(eyes);
      yaw = Math.atan2(to.x, to.z) / DEG;
      pitch = Math.atan2(to.y, Math.hypot(to.x, to.z)) / DEG;
    }
    if (this.options.followPointer) {
      yaw += this._pointerSmooth.x * 5;
      pitch += this._pointerSmooth.y * 3;
    }
    const targetYaw = clamp(yaw + gaze.awayYaw, -32, 32);
    const targetPitch = clamp(pitch + gaze.awayPitch, -20, 20);

    gaze.eyeYaw = damp(gaze.eyeYaw, targetYaw, 24, dt);           // eyes jump (saccade)
    gaze.eyePitch = damp(gaze.eyePitch, targetPitch, 24, dt);
    gaze.headYaw = damp(gaze.headYaw, clamp(targetYaw * 0.5, -16, 16), 3.2, dt);   // head follows part of the way, slowly
    gaze.headPitch = damp(gaze.headPitch, clamp(targetPitch * 0.4, -9, 9), 3.2, dt);
  }

  // Head and neck on top of the idle clip.
  _updateHead(dt, jaw) {
    const { Head: head, Neck: neck, Spine2: chest } = this._bones;
    if (!head) return;
    const style = STATES[this.state];
    const t = this._time;
    const amount = style.motion;
    const nod = Math.sin(this._emphasis.value * Math.PI) * 2.4;
    const talk = this._speech ? jaw * 1.2 : 0;

    const yaw = this._gaze.headYaw + drift(t * 0.9, 1) * 1.6 * amount;
    const pitch = this._gaze.headPitch + drift(t * 1.1, 2) * 1.1 * amount - nod - talk;
    const roll = style.tilt + drift(t * 0.7, 3) * 1.0 * amount;
    this._tilt = damp(this._tilt === undefined ? roll : this._tilt, roll, 3, dt);

    this._restIfNotAnimated(neck);
    this._restIfNotAnimated(head);
    if (chest && this.state === 'listening') this._addWorldRotation(chest, -1.2, 0, 0);
    if (neck) this._addWorldRotation(neck, pitch * 0.35, yaw * 0.35, this._tilt * 0.4);
    this._addWorldRotation(head, pitch * 0.65, yaw * 0.65, this._tilt * 0.6);
  }

  _restIfNotAnimated(bone) {
    if (bone && !this._animated.has(bone.name)) bone.quaternion.copy(this._rest.get(bone));
  }

  // Rotate a bone by world-axis angles (degrees): pitch up, yaw to the character's left, roll.
  _addWorldRotation(bone, pitchUp, yawLeft, roll) {
    const { q1, q2, q3, e } = this._tmp;
    e.set(-pitchUp * DEG, yawLeft * DEG, roll * DEG, 'YXZ');
    q1.setFromEuler(e);
    bone.parent.updateWorldMatrix(true, false);
    bone.parent.getWorldQuaternion(q2);
    q3.copy(q2).invert().multiply(q1).multiply(q2);
    bone.quaternion.premultiply(q3);
  }

  // Blendshapes: blink + eye direction + mood + lip sync, summed and clamped.
  _updateFace(dt) {
    const weights = this._weights;
    for (const key of this._keys) weights[key] = 0;
    const style = STATES[this.state];
    const t = this._time;

    // mood and state, eased so changes of expression take about half a second
    const target = addShape(addShape({}, MOODS[this.mood]), style.face);
    target.mouthSmileLeft = (target.mouthSmileLeft || 0) + drift(t * 0.5, 5) * 0.025;
    target.mouthSmileRight = (target.mouthSmileRight || 0) + drift(t * 0.5, 6) * 0.025;
    target.browInnerUp = (target.browInnerUp || 0) + Math.max(0, drift(t * 0.4, 7)) * 0.05 + Math.sin(this._emphasis.value * Math.PI) * 0.22;
    target.browOuterUpLeft = (target.browOuterUpLeft || 0) + Math.sin(this._emphasis.value * Math.PI) * 0.14;
    target.browOuterUpRight = (target.browOuterUpRight || 0) + Math.sin(this._emphasis.value * Math.PI) * 0.14;
    for (const key of new Set([...Object.keys(target), ...Object.keys(this._expression)])) {
      this._expression[key] = damp(this._expression[key] || 0, target[key] || 0, 5, dt);
      weights[key] = (weights[key] || 0) + this._expression[key];
    }

    // blink
    const blink = this._blink;
    blink.timer -= dt;
    if (blink.timer <= 0 && blink.phase < 0) {
      blink.phase = 0;
      blink.again = Math.random() < 0.12;
    }
    if (blink.phase >= 0) {
      blink.phase += dt;
      const close = 0.07, hold = 0.035, open = 0.15;
      let v = 0;
      if (blink.phase < close) v = blink.phase / close;
      else if (blink.phase < close + hold) v = 1;
      else if (blink.phase < close + hold + open) v = 1 - (blink.phase - close - hold) / open;
      else {
        blink.phase = -1;
        blink.timer = blink.again ? 0.1 : rand(...style.blink);
        blink.again = false;
      }
      blink.value = v * v * (3 - 2 * v);
    } else {
      blink.value = 0;
    }
    weights.eyeBlinkLeft = (weights.eyeBlinkLeft || 0) + blink.value;
    weights.eyeBlinkRight = (weights.eyeBlinkRight || 0) + blink.value;

    // eyes: whatever the head has not turned yet, the eyes cover
    const gaze = this._gaze;
    const yaw = clamp((gaze.eyeYaw - gaze.headYaw) / 30, -1, 1);
    const pitch = clamp((gaze.eyePitch - gaze.headPitch) / 24, -1, 1);
    if (yaw > 0) { weights.eyeLookOutLeft = yaw; weights.eyeLookInRight = yaw; }
    else { weights.eyeLookInLeft = -yaw; weights.eyeLookOutRight = -yaw; }
    if (pitch > 0) { weights.eyeLookUpLeft = pitch; weights.eyeLookUpRight = pitch; }
    else { weights.eyeLookDownLeft = -pitch; weights.eyeLookDownRight = -pitch; }

    // lip sync on top; a talking mouth carries less of the resting smile
    const talking = Math.min(1, (this._viseme.jawOpen || 0) * 3 + (this._viseme.mouthPucker || 0) * 2);
    weights.mouthSmileLeft *= 1 - 0.5 * talking;
    weights.mouthSmileRight *= 1 - 0.5 * talking;
    for (const key in this._viseme) weights[key] = (weights[key] || 0) + this._viseme[key];

    if (this._manual) {
      for (const key of this._keys) weights[key] = 0;
      Object.assign(weights, this._manual);
    }
    for (const mesh of this._faceMeshes) {
      const dictionary = mesh.morphTargetDictionary;
      const influences = mesh.morphTargetInfluences;
      for (const key in dictionary) influences[dictionary[key]] = clamp(weights[key] || 0, 0, 1);
    }
  }

  _updateCamera(dt) {
    const [bottom, topFraction, width] = FRAMINGS[this.framing] || FRAMINGS.medium;
    const yMin = bottom * this.height;
    const yMax = topFraction * this.height;
    const halfFov = (this.camera.fov * DEG) / 2;
    const viewHeight = Math.max(1, this.container.clientHeight);
    const insets = this._insets || { top: 0, bottom: 0 };
    const free = clamp((viewHeight - insets.top - insets.bottom) / viewHeight, 0.35, 1);
    const fitHeight = (yMax - yMin) / 2 / Math.tan(halfFov) / free;
    const fitWidth = width / 2 / (Math.tan(halfFov) * this.camera.aspect);
    const distance = Math.max(fitHeight, fitWidth) / 0.92;
    // Centre the framing in the free band, not in the whole canvas.
    const metresPerPixel = (2 * distance * Math.tan(halfFov)) / viewHeight;
    const shift = ((insets.top - insets.bottom) / 2) * metresPerPixel;

    this._pointerSmooth.x = damp(this._pointerSmooth.x, this._pointer.x, 3, dt);
    this._pointerSmooth.y = damp(this._pointerSmooth.y, this._pointer.y, 3, dt);
    const follow = this.options.followPointer ? 1 : 0;
    const orbit = (this.orbit + this._pointerSmooth.x * 3.5 * follow + drift(this._time * 0.25, 9) * 0.5) * DEG;
    const lift = (this._pointerSmooth.y * 1.5 * follow + drift(this._time * 0.21, 10) * 0.3) * DEG;

    const target = this._tmp.v1.set(0, (yMin + yMax) / 2 + shift, 0);
    const position = this._tmp.v2.set(
      Math.sin(orbit) * distance,
      target.y + Math.sin(lift) * distance + distance * 0.02,
      Math.cos(orbit) * distance,
    );
    if (!this._cam.ready) {
      this._cam.position.copy(position);
      this._cam.target.copy(target);
      this._cam.ready = true;
    }
    const k = 1 - Math.exp(-3.2 * dt);
    this._cam.position.lerp(position, k);
    this._cam.target.lerp(target, k);
    this.camera.position.copy(this._cam.position);
    this.camera.lookAt(this._cam.target);
  }
}
