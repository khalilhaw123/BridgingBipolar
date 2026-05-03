/**
 * Under-16 companion scene.
 * Tries to load GLB/GLTF avatar first, falls back to procedural placeholder.
 * Exports: createUnder16Avatar(canvas, options)
 */
import * as THREE from "three";
import { GLTFLoader } from "https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";

/**
 * @param {HTMLCanvasElement} canvas
 * @param {{ modelCandidates?: string[] }} [options]
 * @returns {{ setMode: (m: 'idle'|'listen'|'speak') => void, setMouth: (t: number) => void, dispose: () => void }}
 */
export function createUnder16Avatar(canvas, options = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 1.35, 2.6);
  camera.lookAt(0, 1.1, 0);

  scene.add(new THREE.HemisphereLight(0xb8c8e8, 0x1a1a22, 0.85));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(2.2, 4.5, 3.0);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x6eb8ff, 0.35);
  rim.position.set(-3, 2, -2);
  scene.add(rim);

  const root = new THREE.Group();
  scene.add(root);

  const runtime = {
    placeholderParts: null,
    avatarModel: null,
    jawBone: null,
    headBone: null,
    mouthMesh: null,
    mouthMorphIndex: -1,
  };

  function buildPlaceholder() {
    const skin = new THREE.MeshStandardMaterial({
      color: 0xd8c4b4,
      roughness: 0.55,
      metalness: 0.05,
    });
    const shirt = new THREE.MeshStandardMaterial({
      color: 0x3a6fa8,
      roughness: 0.65,
      metalness: 0.02,
    });

    const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.38, 0.75, 8, 16), shirt);
    body.position.y = 0.95;
    root.add(body);

    const head = new THREE.Group();
    head.position.set(0, 1.62, 0);
    root.add(head);

    const skull = new THREE.Mesh(new THREE.SphereGeometry(0.28, 32, 24), skin);
    head.add(skull);

    const jaw = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.1, 0.22), skin);
    jaw.position.set(0, -0.16, 0.06);
    jaw.rotation.set(0, 0, 0);
    jaw.geometry.translate(0, -0.05, 0.04);
    head.add(jaw);

    runtime.placeholderParts = { skin, shirt, body, head, skull, jaw };
  }

  function findJawAndHead(modelRoot) {
    let jawBone = null;
    let headBone = null;
    let mouthMesh = null;
    let mouthMorphIndex = -1;

    modelRoot.traverse((obj) => {
      const name = (obj.name || "").toLowerCase();
      if (!headBone && obj.isBone && (name.includes("head") || name.includes("neck"))) {
        headBone = obj;
      }
      if (!jawBone && obj.isBone && (name.includes("jaw") || name.includes("mouth"))) {
        jawBone = obj;
      }
      if (!mouthMesh && obj.isMesh && obj.morphTargetDictionary) {
        const keys = Object.keys(obj.morphTargetDictionary);
        const k = keys.find((x) => /jawopen|mouthopen|openmouth|viseme_aa/i.test(x));
        if (k) {
          mouthMesh = obj;
          mouthMorphIndex = obj.morphTargetDictionary[k];
        }
      }
    });

    return { jawBone, headBone, mouthMesh, mouthMorphIndex };
  }

  function applyModelScaleAndPose(model) {
    model.updateWorldMatrix(true, true);
    const box = new THREE.Box3().setFromObject(model);
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    const h = Math.max(0.001, size.y);
    const targetH = 1.7;
    const scale = targetH / h;
    model.scale.setScalar(scale);
    model.position.sub(center.multiplyScalar(scale));
    model.position.y += 0.95;
  }

  async function tryLoadAvatarModel() {
    const loader = new GLTFLoader();
    const candidates = [
      ...(Array.isArray(options.modelCandidates) ? options.modelCandidates : []),
      "/static/models/avatar.glb",
      "/static/models/avatar.gltf",
      "/static/avatar.glb",
      "/static/avatar.gltf",
    ];
    for (const url of candidates) {
      try {
        const gltf = await loader.loadAsync(url);
        const model = gltf.scene || gltf.scenes?.[0];
        if (!model) continue;
        applyModelScaleAndPose(model);
        root.add(model);
        runtime.avatarModel = model;
        const found = findJawAndHead(model);
        runtime.jawBone = found.jawBone;
        runtime.headBone = found.headBone;
        runtime.mouthMesh = found.mouthMesh;
        runtime.mouthMorphIndex = found.mouthMorphIndex;
        return;
      } catch (_) {
        // Try next candidate silently.
      }
    }
    buildPlaceholder();
  }

  let mode = "idle";
  let mouthOpen = 0;
  let t = 0;
  let disposed = false;
  let raf = 0;

  function resize() {
    const w = canvas.clientWidth || 1;
    const h = canvas.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  resize();
  const ro = new ResizeObserver(() => resize());
  ro.observe(canvas.parentElement || canvas);

  function tick() {
    if (disposed) return;
    raf = requestAnimationFrame(tick);
    const dt = 1 / 60;
    t += dt;

    const breath = Math.sin(t * 2.1) * 0.012;
    if (runtime.placeholderParts) {
      const { body, head } = runtime.placeholderParts;
      body.scale.set(1, 1 + breath * 6, 1);
      head.rotation.y = Math.sin(t * 0.9) * 0.06;
      head.rotation.x = Math.sin(t * 0.55) * 0.02;
    } else if (runtime.headBone) {
      runtime.headBone.rotation.y = Math.sin(t * 0.9) * 0.06;
      runtime.headBone.rotation.x = Math.sin(t * 0.55) * 0.02;
    }

    if (mode === "listen") {
      root.rotation.x = THREE.MathUtils.lerp(root.rotation.x, 0.08, 0.08);
    } else {
      root.rotation.x = THREE.MathUtils.lerp(root.rotation.x, 0, 0.08);
    }

    const targetJaw = mode === "speak" ? mouthOpen : 0;
    if (runtime.placeholderParts) {
      const { jaw } = runtime.placeholderParts;
      const jawX = THREE.MathUtils.lerp(jaw.rotation.x, 0.15 + targetJaw * 0.55, 0.22);
      jaw.rotation.x = jawX;
    } else if (runtime.jawBone) {
      const jawX = THREE.MathUtils.lerp(runtime.jawBone.rotation.x, targetJaw * 0.55, 0.3);
      runtime.jawBone.rotation.x = jawX;
    } else if (runtime.mouthMesh && runtime.mouthMorphIndex >= 0 && runtime.mouthMesh.morphTargetInfluences) {
      const arr = runtime.mouthMesh.morphTargetInfluences;
      arr[runtime.mouthMorphIndex] = THREE.MathUtils.lerp(arr[runtime.mouthMorphIndex] || 0, targetJaw, 0.35);
    }
    renderer.render(scene, camera);
  }
  void tryLoadAvatarModel().finally(() => {
    tick();
  });

  return {
    setMode(m) {
      if (m === "idle" || m === "listen" || m === "speak") mode = m;
    },
    setMouth(v) {
      mouthOpen = Math.max(0, Math.min(1, v));
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.dispose();
      if (runtime.placeholderParts) {
        const { skull, jaw, body, skin, shirt } = runtime.placeholderParts;
        skull.geometry.dispose();
        jaw.geometry.dispose();
        body.geometry.dispose();
        skin.dispose();
        shirt.dispose();
      }
    },
  };
}
