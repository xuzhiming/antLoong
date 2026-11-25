import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const container = document.getElementById("canvas-container");
const loaderOverlay = document.getElementById("loader-overlay");
const loaderTitle = document.getElementById("loader-title");
const loaderBarFill = document.getElementById("loader-bar-fill");
const loaderPercent = document.getElementById("loader-percent");
const errorBanner = document.getElementById("error-banner");
const errorText = document.getElementById("error-text");
const hudTitleEl = document.getElementById("hud-title");
const meshStatsEl = document.getElementById("mesh-stats");
const partChecksEl = document.getElementById("part-checks");
const refImgEl = document.getElementById("ref-img");
const bloomSlider = document.getElementById("bloom-slider");
const bloomVal = document.getElementById("bloom-val");
const exposureSlider = document.getElementById("exposure-slider");
const exposureVal = document.getElementById("exposure-val");
const animControlGroup = document.getElementById("anim-control-group");
const animClipBtns = document.querySelectorAll("#anim-clips .seg-btn");
const btnAnimPlay = document.getElementById("btn-anim-play");

function showError(msg) {
  errorText.textContent = msg;
  errorBanner.classList.remove("hidden");
  loaderOverlay.classList.add("hidden");
}

document.getElementById("error-close").addEventListener("click", () => {
  errorBanner.classList.add("hidden");
});

// Renderer
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
} catch (err) {
  showError("WebGL initialization failed: " + (err?.message || err));
  throw err;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.10;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

// Scene & Environment
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070a10);

const pmrem = new THREE.PMREMGenerator(renderer);
const envRT = pmrem.fromScene(new RoomEnvironment(), 0.04);
scene.environment = envRT.texture;
scene.environmentIntensity = 0.72;

// Camera & OrbitControls
function computeResponsiveFov() {
  const aspect = window.innerWidth / Math.max(1, window.innerHeight);
  return aspect < 1.25 ? Math.min(54, 38 * (1.25 / Math.max(0.75, aspect))) : 38;
}

const camera = new THREE.PerspectiveCamera(computeResponsiveFov(), window.innerWidth / window.innerHeight, 0.1, 120);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 1.1;
controls.maxDistance = 32.0;
controls.minPolarAngle = 0.01;
controls.maxPolarAngle = Math.PI - 0.01;

// Postprocessing (UnrealBloomPass)
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
const bloomPass = new UnrealBloomPass(
  new THREE.Vector2(window.innerWidth, window.innerHeight),
  0.22, // strength
  0.44, // radius
  0.94  // threshold
);
const outputPass = new OutputPass();
composer.addPass(renderPass);
composer.addPass(bloomPass);
composer.addPass(outputPass);

// Lighting Rig
const hemiLight = new THREE.HemisphereLight(0xeef4ff, 0x1c182c, 0.65);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight(0xfff5ec, 2.0);
keyLight.position.set(-6.5, 9.0, 9.0);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x88e8ff, 1.05);
fillLight.position.set(8.5, 4.5, 6.0);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xe6c8ff, 1.45);
rimLight.position.set(2.5, 8.5, -8.5);
scene.add(rimLight);

const underLight = new THREE.DirectionalLight(0xe6eeff, 0.90);
underLight.position.set(0.0, -8.0, 1.0);
scene.add(underLight);

const LIGHTING_PRESETS = {
  cyberpunk: {
    bg: 0x070a10,
    envInt: 0.72,
    hemi: [0xeef4ff, 0x1c182c, 0.65],
    key: [0xfff5ec, 2.0],
    fill: [0x88e8ff, 1.05],
    rim: [0xe6c8ff, 1.45],
    under: [0xe6eeff, 0.90],
  },
  studio: {
    bg: 0xe8ecf2,
    envInt: 0.95,
    hemi: [0xffffff, 0x8a94a6, 0.95],
    key: [0xffffff, 2.2],
    fill: [0xdde9ff, 1.3],
    rim: [0xf4f7fc, 1.35],
    under: [0xe8eef5, 1.1],
  },
  dark_rim: {
    bg: 0x030407,
    envInt: 0.35,
    hemi: [0x9bb8ff, 0x090612, 0.25],
    key: [0xffe7dc, 1.1],
    fill: [0x00e8ff, 1.35],
    rim: [0xff4da6, 2.2],
    under: [0x506880, 0.45],
  },
};

function applyLightingPreset(name) {
  const p = LIGHTING_PRESETS[name];
  if (!p) return;
  const lightMul = activeModelKey === "crystal" ? (name === "studio" ? 0.68 : 0.65) : 1.0;
  scene.background = new THREE.Color(p.bg);
  scene.environmentIntensity = p.envInt * (activeModelKey === "crystal" ? 0.68 : 1.0);
  hemiLight.color.setHex(p.hemi[0]);
  hemiLight.groundColor.setHex(p.hemi[1]);
  hemiLight.intensity = p.hemi[2] * lightMul;
  keyLight.color.setHex(p.key[0]);
  keyLight.intensity = p.key[1] * lightMul;
  fillLight.color.setHex(p.fill[0]);
  fillLight.intensity = p.fill[1] * lightMul;
  rimLight.color.setHex(p.rim[0]);
  rimLight.intensity = p.rim[1] * lightMul;
  underLight.color.setHex(p.under[0]);
  underLight.intensity = p.under[1] * lightMul;
  const isLight = name === "studio";
  const mats = Array.isArray(gridHelper.material) ? gridHelper.material : [gridHelper.material];
  mats.forEach((m) => {
    m.opacity = isLight ? 0.16 : 0.28;
  });
}

// Cyber Floor Grid + Holographic Pedestal Rings
const floorGroup = new THREE.Group();
const gridHelper = new THREE.GridHelper(18, 36, 0x40f4ff, 0x1f2842);
gridHelper.position.y = -0.004;
if (Array.isArray(gridHelper.material)) {
  gridHelper.material.forEach((m) => {
    m.transparent = true;
    m.opacity = 0.28;
  });
} else {
  gridHelper.material.transparent = true;
  gridHelper.material.opacity = 0.28;
}
floorGroup.add(gridHelper);

function createRing(radius, thickness, colorHex, opacity) {
  const geo = new THREE.RingGeometry(radius - thickness, radius, 96);
  const mat = new THREE.MeshBasicMaterial({
    color: colorHex,
    side: THREE.DoubleSide,
    transparent: true,
    opacity,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.x = -Math.PI * 0.5;
  mesh.position.y = -0.002;
  return mesh;
}
const ringInner = createRing(1.0, 0.005, 0x40f4ff, 0.48);
const ringOuter = createRing(1.06, 0.003, 0xc8a4ff, 0.36);
floorGroup.add(ringInner);
floorGroup.add(ringOuter);
scene.add(floorGroup);

// Dual-model Configuration
const MODELS = {
  crystal: {
    title: "CRYSTAL DRAGON",
    glb: "./assets/crystal_dragon.glb",
    ref: "./assets/crystal_reference_6view.jpg",
    ringScale: 2.9,
    bloom: { strength: 0.18, threshold: 1.45, exposure: 0.94 },
    presets: {
      hero:   { pos: new THREE.Vector3(-5.6,  3.1,  6.8), target: new THREE.Vector3(0, 1.50, -0.05) },
      head:   { pos: new THREE.Vector3( 1.0,  1.85, 3.2), target: new THREE.Vector3(0, 1.82,  1.15) },
      front:  { pos: new THREE.Vector3( 0.0,  1.25, 5.6), target: new THREE.Vector3(0, 1.20, -0.2) },
      rear:   { pos: new THREE.Vector3( 0.0,  1.25,-5.8), target: new THREE.Vector3(0, 1.20, -0.2) },
      left:   { pos: new THREE.Vector3( 6.4,  1.25,-0.2), target: new THREE.Vector3(0, 1.20, -0.2) },
      right:  { pos: new THREE.Vector3(-6.4,  1.25,-0.2), target: new THREE.Vector3(0, 1.20, -0.2) },
      top:    { pos: new THREE.Vector3( 0.0,  6.8, -0.2), target: new THREE.Vector3(0, 1.20, -0.2) },
      bottom: { pos: new THREE.Vector3( 0.0, -6.5, -0.2), target: new THREE.Vector3(0, 1.20, -0.2) },
    },
    parts: [
      { key: "head",     label: "Head & Horns",   regex: /Head|Eyes|Horns/i },
      { key: "wings",    label: "Wings",          regex: /Wings/i },
      { key: "armor",    label: "Shoulder Armor", regex: /Pauldrons/i },
      { key: "body",     label: "Body & Spines",  regex: /Body|Dorsal_Spines/i },
      { key: "belly",    label: "Belly & Tail",   regex: /Belly_Scutes|Tail_Plume/i },
      { key: "limbs",    label: "Limbs & Paws",   regex: /Limbs|Claws/i },
    ],
  },
  cyberpunk: {
    title: "CYBERPUNK DRAGON",
    glb: "./assets/cyberpunk_dragon.glb",
    ref: "./assets/reference_6view.jpg",
    ringScale: 5.8,
    bloom: { strength: 0.28, threshold: 0.92, exposure: 1.15 },
    presets: {
      hero:   { pos: new THREE.Vector3(-9.2,  4.3, 11.2), target: new THREE.Vector3(0, 1.55, 0.05) },
      head:   { pos: new THREE.Vector3( 2.1,  2.5,  6.4), target: new THREE.Vector3(0, 2.00, 3.85) },
      front:  { pos: new THREE.Vector3( 0.0,  1.55, 9.2), target: new THREE.Vector3(0, 1.55, 0.50) },
      rear:   { pos: new THREE.Vector3( 0.0,  1.55,-9.2), target: new THREE.Vector3(0, 1.55,-0.50) },
      left:   { pos: new THREE.Vector3(14.6,  1.55, 0.0), target: new THREE.Vector3(0, 1.55, 0.00) },
      right:  { pos: new THREE.Vector3(-14.6, 1.55, 0.0), target: new THREE.Vector3(0, 1.55, 0.00) },
      top:    { pos: new THREE.Vector3( 0.0, 15.0,  0.05), target: new THREE.Vector3(0, 1.55, 0.00) },
      bottom: { pos: new THREE.Vector3( 0.0,-14.0,  0.05), target: new THREE.Vector3(0, 1.55, 0.00) },
    },
    parts: [
      { key: "head",     label: "Head & Antlers", regex: /Head|Eyes|Antlers|Beard|Whiskers/i },
      { key: "mane",     label: "Mane & Crest",   regex: /Mane|Dorsal_Crest|Tail_Plume|Limb_Tufts/i },
      { key: "conduits", label: "Cyan Conduits",  regex: /Cyan_Conduits/i },
      { key: "scutes",   label: "Belly Scutes",   regex: /Belly_Scutes/i },
      { key: "body",     label: "Body Scales",    regex: /Body/i },
      { key: "limbs",    label: "Limbs & Claws",  regex: /Limbs|Claws/i },
    ],
  },
};

let activeModelKey = "crystal";
let activeRoot = null;
let partBuckets = {};
let allMeshes = [];
let currentView = "hero";
let userGridEnabled = true;
let wireframeOn = false;
let camAnim = null;
let mixer = null;
let animActions = {};
let activeClipName = "Takeoff_And_Fly";
let animPlaying = true;
let lastFrameTime = performance.now();

function syncFloorVisibility() {
  floorGroup.visible = userGridEnabled && currentView !== "bottom";
}

function setCameraView(viewName, instant = false) {
  const presets = MODELS[activeModelKey].presets;
  const preset = presets[viewName];
  if (!preset) return;
  currentView = viewName;
  syncFloorVisibility();

  document.querySelectorAll(".cam-btn[data-view]").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === viewName);
  });

  if (instant) {
    camera.position.copy(preset.pos);
    controls.target.copy(preset.target);
    controls.update();
    camAnim = null;
  } else {
    camAnim = {
      startPos: camera.position.clone(),
      endPos: preset.pos.clone(),
      startTarget: controls.target.clone(),
      endTarget: preset.target.clone(),
      startTime: performance.now(),
      duration: 580,
    };
  }
}

function rebuildPartCheckboxes(modelCfg) {
  partChecksEl.innerHTML = "";
  partBuckets = {};
  modelCfg.parts.forEach((p) => {
    partBuckets[p.key] = [];
    const lbl = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = true;
    cb.dataset.part = p.key;
    cb.addEventListener("change", () => {
      (partBuckets[p.key] || []).forEach((obj) => {
        obj.visible = cb.checked;
      });
    });
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(" " + p.label));
    partChecksEl.appendChild(lbl);
  });
}

function disposeModel(root) {
  if (!root) return;
  scene.remove(root);
  root.traverse((obj) => {
    if (obj.isMesh) {
      if (obj.geometry) obj.geometry.dispose();
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      mats.forEach((m) => m && m.dispose());
    }
  });
}

function resetSkinnedRestPose() {
  if (!activeRoot) return;
  activeRoot.traverse((obj) => {
    if (obj.isSkinnedMesh && obj.skeleton) {
      obj.skeleton.pose();
    }
  });
}

function setAnimationPlaying(playing) {
  animPlaying = Boolean(playing);
  btnAnimPlay.classList.toggle("active", animPlaying);
  btnAnimPlay.textContent = animPlaying ? "Pause Flight" : "Play Flight";
}

function setAnimationClip(clipName) {
  activeClipName = clipName;
  animClipBtns.forEach((b) => {
    b.classList.toggle("active", b.dataset.clip === clipName);
  });
  if (!mixer) return;
  mixer.stopAllAction();
  if (clipName === "rest") {
    resetSkinnedRestPose();
    setAnimationPlaying(false);
    return;
  }
  const action = animActions[clipName] || Object.values(animActions)[0];
  if (action) {
    action.reset();
    action.setLoop(THREE.LoopRepeat, Infinity);
    action.play();
    mixer.update(0);
    setAnimationPlaying(true);
  }
}

const loader = new GLTFLoader();
let loadToken = 0;

function loadDragonModel(modelKey) {
  const cfg = MODELS[modelKey];
  if (!cfg) return;
  activeModelKey = modelKey;
  const myToken = ++loadToken;
  if (mixer) {
    mixer.stopAllAction();
    mixer = null;
  }
  animActions = {};
  animControlGroup.classList.add("hidden");

  document.querySelectorAll(".model-btn[data-model]").forEach((b) => {
    b.classList.toggle("active", b.dataset.model === modelKey);
  });

  hudTitleEl.textContent = cfg.title;
  refImgEl.src = cfg.ref;
  ringInner.scale.setScalar(cfg.ringScale);
  ringOuter.scale.setScalar(cfg.ringScale);

  const activePresetBtn = document.querySelector("#lighting-presets .seg-btn.active");
  applyLightingPreset(activePresetBtn ? activePresetBtn.dataset.preset : "cyberpunk");

  bloomPass.strength = cfg.bloom.strength;
  bloomPass.threshold = cfg.bloom.threshold;
  bloomSlider.value = String(cfg.bloom.strength);
  bloomVal.textContent = cfg.bloom.strength.toFixed(2);
  renderer.toneMappingExposure = cfg.bloom.exposure;
  exposureSlider.value = String(cfg.bloom.exposure);
  exposureVal.textContent = cfg.bloom.exposure.toFixed(2);

  rebuildPartCheckboxes(cfg);
  setCameraView("hero", true);

  loaderTitle.textContent = `LOADING ${cfg.title}`;
  loaderBarFill.style.width = "0%";
  loaderPercent.textContent = "0%";
  loaderOverlay.classList.remove("hidden");
  loaderOverlay.style.opacity = "1";

  loader.load(
    cfg.glb + "?v=20261125b",
    (gltf) => {
      if (myToken !== loadToken) return;
      disposeModel(activeRoot);
      const model = gltf.scene;
      activeRoot = model;
      allMeshes = [];

      let partCount = 0;
      let triCount = 0;
      model.traverse((obj) => {
        if (obj.isBone) return;
        const name = obj.name || "";
        let matchedKey = null;
        for (const p of cfg.parts) {
          if (p.regex.test(name)) {
            matchedKey = p.key;
            break;
          }
        }
        if (matchedKey && !obj.isMesh) {
          partBuckets[matchedKey].push(obj);
          partCount += 1;
        }
        if (obj.isMesh) {
          if (obj.isSkinnedMesh) {
            obj.frustumCulled = false;
          }
          allMeshes.push(obj);
          if (matchedKey) partBuckets[matchedKey].push(obj);
          const geom = obj.geometry;
          if (geom) {
            triCount += geom.index ? geom.index.count / 3 : geom.attributes.position.count / 3;
          }
          const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
          mats.forEach((m) => {
            if (m) {
              m.wireframe = wireframeOn;
              m.needsUpdate = true;
            }
          });
        }
      });

      scene.add(model);

      if (gltf.animations && gltf.animations.length > 0) {
        mixer = new THREE.AnimationMixer(model);
        gltf.animations.forEach((clip) => {
          animActions[clip.name] = mixer.clipAction(clip);
        });
        animControlGroup.classList.remove("hidden");
        const defaultClip = animActions.Takeoff_And_Fly ? "Takeoff_And_Fly" : gltf.animations[0].name;
        setAnimationClip(defaultClip);
      } else {
        mixer = null;
        animActions = {};
        animControlGroup.classList.add("hidden");
      }

      const box = new THREE.Box3().setFromObject(model);
      const size = new THREE.Vector3();
      box.getSize(size);
      meshStatsEl.textContent =
        `${partCount || allMeshes.length} Parts · ${Math.round(triCount).toLocaleString()} Tris · ` +
        `${size.x.toFixed(2)} × ${size.z.toFixed(2)} × ${size.y.toFixed(2)} m`;

      loaderBarFill.style.width = "100%";
      loaderPercent.textContent = "100%";
      setTimeout(() => {
        if (myToken !== loadToken) return;
        loaderOverlay.style.opacity = "0";
        setTimeout(() => loaderOverlay.classList.add("hidden"), 350);
      }, 120);
    },
    (xhr) => {
      if (myToken !== loadToken) return;
      if (xhr.total > 0) {
        const pct = Math.min(99, Math.round((xhr.loaded / xhr.total) * 100));
        loaderBarFill.style.width = `${pct}%`;
        loaderPercent.textContent = `${pct}%`;
      }
    },
    (err) => {
      if (myToken !== loadToken) return;
      showError(`Failed to load ${cfg.glb}: ` + (err?.message || err));
    }
  );
}

// UI Bindings
const controlsPanel = document.getElementById("controls-panel");
const togglePanelBtn = document.getElementById("toggle-panel-btn");
if (window.innerWidth < 1000) {
  controlsPanel.classList.add("collapsed");
  togglePanelBtn.textContent = "Expand";
}
togglePanelBtn.addEventListener("click", () => {
  const collapsed = controlsPanel.classList.toggle("collapsed");
  togglePanelBtn.textContent = collapsed ? "Expand" : "Collapse";
});

document.querySelectorAll(".model-btn[data-model]").forEach((btn) => {
  btn.addEventListener("click", () => {
    if (btn.dataset.model !== activeModelKey) {
      loadDragonModel(btn.dataset.model);
    }
  });
});

document.querySelectorAll("#lighting-presets .seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#lighting-presets .seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    applyLightingPreset(btn.dataset.preset);
  });
});

animClipBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    setAnimationClip(btn.dataset.clip);
  });
});

btnAnimPlay.addEventListener("click", () => {
  if (!mixer) return;
  if (activeClipName === "rest") {
    setAnimationClip(animActions.Takeoff_And_Fly ? "Takeoff_And_Fly" : Object.keys(animActions)[0]);
  } else {
    setAnimationPlaying(!animPlaying);
  }
});

bloomSlider.addEventListener("input", () => {
  const v = parseFloat(bloomSlider.value);
  bloomPass.strength = v;
  bloomVal.textContent = v.toFixed(2);
});

exposureSlider.addEventListener("input", () => {
  const v = parseFloat(exposureSlider.value);
  renderer.toneMappingExposure = v;
  exposureVal.textContent = v.toFixed(2);
});

const btnAutorotate = document.getElementById("btn-autorotate");
btnAutorotate.addEventListener("click", () => {
  controls.autoRotate = !controls.autoRotate;
  controls.autoRotateSpeed = 1.6;
  btnAutorotate.classList.toggle("active", controls.autoRotate);
});

const btnWireframe = document.getElementById("btn-wireframe");
btnWireframe.addEventListener("click", () => {
  wireframeOn = !wireframeOn;
  btnWireframe.classList.toggle("active", wireframeOn);
  allMeshes.forEach((mesh) => {
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    mats.forEach((m) => {
      if (m) m.wireframe = wireframeOn;
    });
  });
});

const btnGrid = document.getElementById("btn-grid");
btnGrid.addEventListener("click", () => {
  userGridEnabled = !userGridEnabled;
  btnGrid.classList.toggle("active", userGridEnabled);
  syncFloorVisibility();
});

const refModal = document.getElementById("ref-modal");
const btnReference = document.getElementById("btn-reference");
const refCloseBtn = document.getElementById("ref-close-btn");
function toggleRefModal(force) {
  const show = typeof force === "boolean" ? force : refModal.classList.contains("hidden");
  refModal.classList.toggle("hidden", !show);
  btnReference.classList.toggle("active", show);
}
btnReference.addEventListener("click", () => toggleRefModal());
refCloseBtn.addEventListener("click", () => toggleRefModal(false));

const refHeader = document.getElementById("ref-modal-header");
let dragState = null;
refHeader.addEventListener("pointerdown", (e) => {
  if (e.target === refCloseBtn) return;
  const rect = refModal.getBoundingClientRect();
  dragState = { dx: e.clientX - rect.left, dy: e.clientY - rect.top };
  refHeader.setPointerCapture(e.pointerId);
});
refHeader.addEventListener("pointermove", (e) => {
  if (!dragState) return;
  const maxX = Math.max(8, window.innerWidth - refModal.offsetWidth - 8);
  const maxY = Math.max(8, window.innerHeight - refModal.offsetHeight - 8);
  refModal.style.left = `${Math.max(8, Math.min(maxX, e.clientX - dragState.dx))}px`;
  refModal.style.top = `${Math.max(8, Math.min(maxY, e.clientY - dragState.dy))}px`;
  refModal.style.bottom = "auto";
});
refHeader.addEventListener("pointerup", () => {
  dragState = null;
});

document.querySelectorAll(".cam-btn[data-view]").forEach((btn) => {
  btn.addEventListener("click", () => setCameraView(btn.dataset.view));
});
document.getElementById("btn-reset").addEventListener("click", () => setCameraView("hero"));

const KEY_VIEW_MAP = {
  "1": "hero",
  "2": "head",
  "3": "front",
  "4": "rear",
  "5": "left",
  "6": "right",
  "7": "top",
  "8": "bottom",
};
window.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  if (e.code === "Space" && mixer) {
    e.preventDefault();
    btnAnimPlay.click();
  } else if (KEY_VIEW_MAP[e.key]) {
    setCameraView(KEY_VIEW_MAP[e.key]);
  } else if (e.key === "r" || e.key === "R") {
    setCameraView("hero");
  }
});

controls.addEventListener("start", () => {
  camAnim = null;
});

window.addEventListener("resize", () => {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.fov = computeResponsiveFov();
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  composer.setSize(w, h);
  bloomPass.setSize(w, h);
});

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.05, Math.max(0.0, (now - lastFrameTime) / 1000.0));
  lastFrameTime = now;
  if (mixer && animPlaying && activeClipName !== "rest") {
    mixer.update(dt);
  }
  if (camAnim) {
    const rawT = Math.min(1.0, (now - camAnim.startTime) / camAnim.duration);
    const ease = 1.0 - Math.pow(1.0 - rawT, 3.0);
    camera.position.lerpVectors(camAnim.startPos, camAnim.endPos, ease);
    controls.target.lerpVectors(camAnim.startTarget, camAnim.endTarget, ease);
    if (rawT >= 1.0) camAnim = null;
  }
  controls.update();
  composer.render();
}

window.__dragonViewer = {
  getState: () => ({
    activeModelKey,
    activeClipName,
    animPlaying,
    clips: Object.keys(animActions),
    skinnedMeshes: allMeshes.filter((m) => m.isSkinnedMesh).length,
  }),
  setClip: (name) => setAnimationClip(name),
  setPlaying: (p) => setAnimationPlaying(p),
};

// Default load: Crystal Baby Dragon
loadDragonModel("crystal");
requestAnimationFrame(animate);
