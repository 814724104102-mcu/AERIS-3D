import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/**
 * AERIS-3D Three.js Viewer — Prompt 3B
 *
 * Modes:
 *   ORBIT  — OrbitControls (default inspect mode, drag/scroll/pan)
 *   FLY    — PointerLockControls + WASD + mouse-look (first-person)
 *
 * Features:
 *   - Raycasting click-to-inspect on terrain mesh
 *   - Real-time water-plane mesh driven by `floodLevel` prop
 *   - Camera mode toggle (Fly / Orbit)
 *   - Auto-rotate in orbit mode
 */
export default function ThreeViewer({ glbUrl, floodLevel = 0, onMeshClick, survivors }) {
  const mountRef       = useRef(null);
  const rendererRef    = useRef(null);
  const sceneRef       = useRef(null);
  const cameraRef      = useRef(null);
  const orbitRef       = useRef(null);
  const flyRef         = useRef(null);
  const rafRef         = useRef(null);
  const autoRotRef     = useRef(false);
  const keysRef        = useRef({});       // WASD state
  const meshesRef      = useRef([]);       // collidable meshes for raycasting
  const waterPlaneRef  = useRef(null);
  const meshBBoxRef    = useRef(null);     // bounding box of loaded terrain
  const flyModeRef     = useRef(false);    // mirror of flyMode state
  const floodRef       = useRef(floodLevel);

  const [loading,  setLoading]  = useState(true);
  const [loadError, setLoadError] = useState(null);
  const [autoRot,  setAutoRot]  = useState(false);
  const [flyMode,  setFlyMode]  = useState(false);
  const [meshInfo, setMeshInfo] = useState(null);
  const [flyLocked, setFlyLocked] = useState(false);
  const [inspectCard, setInspectCard] = useState(null); // {x, y, data}

  // Keep refs in sync with state
  useEffect(() => { autoRotRef.current = autoRot; }, [autoRot]);
  useEffect(() => { flyModeRef.current = flyMode; }, [flyMode]);
  useEffect(() => { floodRef.current = floodLevel; }, [floodLevel]);

  // ── Scene setup (runs once per glbUrl) ─────────────────────────
  useEffect(() => {
    if (!glbUrl || !mountRef.current) return;

    const container = mountRef.current;
    const W = container.clientWidth  || window.innerWidth;
    const H = container.clientHeight || window.innerHeight;

    // Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#050508');
    scene.fog = new THREE.FogExp2('#050508', 0.04);
    sceneRef.current = scene;

    // Camera
    const camera = new THREE.PerspectiveCamera(55, W / H, 0.001, 500);
    camera.position.set(0, 2.2, 2.8);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    rendererRef.current = renderer;

    while (container.firstChild) container.removeChild(container.firstChild);
    container.appendChild(renderer.domElement);

    // Orbit controls (default)
    const orbit = new OrbitControls(camera, renderer.domElement);
    orbit.enableDamping  = true;
    orbit.dampingFactor  = 0.07;
    orbit.minDistance    = 0.3;
    orbit.maxDistance    = 15;
    orbit.maxPolarAngle  = Math.PI / 1.7;
    orbit.target.set(0, 0, 0);
    orbit.update();
    orbitRef.current = orbit;

    // PointerLock controls (fly mode)
    const fly = new PointerLockControls(camera, renderer.domElement);
    flyRef.current = fly;

    fly.addEventListener('lock',   () => setFlyLocked(true));
    fly.addEventListener('unlock', () => setFlyLocked(false));

    // Grid
    const gridHelper = new THREE.GridHelper(5, 24, 0x4ecdc415, 0x4ecdc410);
    gridHelper.position.y = -0.005;
    scene.add(gridHelper);

    // ── Water plane (flood) ──────────────────────────────────────
    const waterGeo  = new THREE.PlaneGeometry(8, 8);
    const waterMat  = new THREE.MeshBasicMaterial({
      color: 0x1a6bff,
      transparent: true,
      opacity: 0.38,
      side: THREE.DoubleSide,
    });
    const waterPlane = new THREE.Mesh(waterGeo, waterMat);
    waterPlane.rotation.x = -Math.PI / 2;
    waterPlane.visible = false;
    waterPlane.renderOrder = 1;
    scene.add(waterPlane);
    waterPlaneRef.current = waterPlane;

    // ── Load GLB ────────────────────────────────────────────────
    const loader = new GLTFLoader();
    loader.load(
      glbUrl,
      (gltf) => {
        const model = gltf.scene;
        let totalVerts = 0;
        let totalFaces = 0;
        const collidable = [];

        model.traverse(node => {
          if (!node.isMesh) return;
          totalVerts += node.geometry.attributes.position?.count || 0;
          totalFaces += (node.geometry.index
            ? node.geometry.index.count
            : node.geometry.attributes.position?.count) / 3 | 0;

          if (node.geometry.hasAttribute('color')) {
            node.material = new THREE.MeshBasicMaterial({
              vertexColors: true,
              side: THREE.DoubleSide,
            });
          } else {
            node.material = new THREE.MeshStandardMaterial({
              color: 0x4ecdc4,
              roughness: 0.8,
              metalness: 0.0,
              side: THREE.DoubleSide,
            });
          }
          collidable.push(node);
        });

        // Center and scale
        const box    = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size   = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale  = 2.0 / Math.max(maxDim, 0.01);

        model.scale.setScalar(scale);
        model.position.copy(center.multiplyScalar(-scale));
        model.rotation.x = -Math.PI / 2;

        const newBox = new THREE.Box3().setFromObject(model);
        gridHelper.position.y = newBox.min.y - 0.01;

        // Position water plane at mesh minimum Y
        waterPlane.position.y = newBox.min.y;
        meshBBoxRef.current = newBox;

        scene.add(model);
        meshesRef.current = collidable;
        setMeshInfo({ verts: totalVerts, faces: totalFaces });
        setLoading(false);
      },
      undefined,
      (err) => {
        console.error('GLB load error:', err);
        setLoadError(err.message || 'Failed to load GLB file');
        setLoading(false);
      }
    );

    // ── WASD key listeners ───────────────────────────────────────
    const onKeyDown = (e) => { keysRef.current[e.code] = true; };
    const onKeyUp   = (e) => { keysRef.current[e.code] = false; };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup',   onKeyUp);

    // ── Raycasting ───────────────────────────────────────────────
    const raycaster = new THREE.Raycaster();
    const mouse     = new THREE.Vector2();

    const onClick = (e) => {
      // Only raycast in orbit mode when not dragging
      if (flyModeRef.current) return;
      const rect = renderer.domElement.getBoundingClientRect();
      mouse.x =  ((e.clientX - rect.left)  / rect.width)  * 2 - 1;
      mouse.y = -((e.clientY - rect.top)   / rect.height) * 2 + 1;

      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(meshesRef.current, true);
      if (hits.length > 0) {
        const hit = hits[0];
        const card = {
          screenX: e.clientX,
          screenY: e.clientY,
          worldPos: hit.point,
        };
        setInspectCard(card);
        if (onMeshClick) onMeshClick(card);
      } else {
        setInspectCard(null);
      }
    };
    renderer.domElement.addEventListener('click', onClick);

    // ── Animate loop ─────────────────────────────────────────────
    const clock = new THREE.Clock();
    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);
      const dt = clock.getDelta();

      if (flyModeRef.current && flyRef.current?.isLocked) {
        // WASD movement
        const speed = 1.8;
        const fly   = flyRef.current;
        if (keysRef.current['KeyW'] || keysRef.current['ArrowUp'])    fly.moveForward( speed * dt);
        if (keysRef.current['KeyS'] || keysRef.current['ArrowDown'])  fly.moveForward(-speed * dt);
        if (keysRef.current['KeyA'] || keysRef.current['ArrowLeft'])  fly.moveRight(-speed * dt);
        if (keysRef.current['KeyD'] || keysRef.current['ArrowRight']) fly.moveRight( speed * dt);
        if (keysRef.current['Space'])     camera.position.y += speed * dt;
        if (keysRef.current['ShiftLeft']) camera.position.y -= speed * dt;
      } else {
        // Orbit mode
        if (orbitRef.current) {
          orbitRef.current.autoRotate      = autoRotRef.current;
          orbitRef.current.autoRotateSpeed = 0.7;
          orbitRef.current.update();
        }
      }

      // Update water plane Y position based on flood level
      if (waterPlaneRef.current && meshBBoxRef.current) {
        const baseY = meshBBoxRef.current.min.y;
        const meshH = meshBBoxRef.current.max.y - meshBBoxRef.current.min.y;
        const scale = meshH > 0 ? meshH / 5.0 : 0.2; // 5m real-world range → mesh units
        const flood = floodRef.current;
        waterPlaneRef.current.visible  = flood > 0;
        waterPlaneRef.current.position.y = baseY + flood * scale;
      }

      renderer.render(scene, camera);
    };
    animate();

    // ── Resize ───────────────────────────────────────────────────
    const onResize = () => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup',   onKeyUp);
      renderer.domElement.removeEventListener('click', onClick);
      cancelAnimationFrame(rafRef.current);
      orbit.dispose();
      fly.dispose();
      renderer.dispose();
    };
  }, [glbUrl]);

  // Toggle fly / orbit mode
  const toggleFlyMode = useCallback(() => {
    setFlyMode(prev => {
      const next = !prev;
      flyModeRef.current = next;
      if (next) {
        // Enable pointer lock
        if (orbitRef.current) orbitRef.current.enabled = false;
        flyRef.current?.lock();
      } else {
        // Return to orbit
        flyRef.current?.unlock();
        if (orbitRef.current) orbitRef.current.enabled = true;
      }
      setAutoRot(false);
      return next;
    });
  }, []);

  const resetCamera = () => {
    if (!cameraRef.current || !orbitRef.current) return;
    cameraRef.current.position.set(0, 2.2, 2.8);
    orbitRef.current.target.set(0, 0, 0);
    orbitRef.current.update();
    setAutoRot(false);
  };

  // ── Best survivor for inspect card ──────────────────────────────
  const matchedSurvivor = useMemo(() => {
    if (!survivors || !inspectCard) return null;
    const hitY = inspectCard.worldPos.y;
    let closest = null;
    let minDiff = Infinity;
    for (const cand of Object.values(survivors)) {
      const diff = Math.abs((cand.terrain_baseline + cand.height_value) - hitY);
      if (diff < minDiff) {
        minDiff = diff;
        closest = cand;
      }
    }
    return closest;
  }, [survivors, inspectCard]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Canvas */}
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {/* Fly mode overlay */}
      {flyMode && !flyLocked && (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'rgba(5,5,8,0.6)', flexDirection: 'column', gap: 12,
          cursor: 'crosshair',
        }} onClick={() => flyRef.current?.lock()}>
          <div style={{ color: '#4ecdc4', fontSize: 13, letterSpacing: '0.2em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
            Click to enter first-person mode
          </div>
          <div style={{ color: 'rgba(78,205,196,0.5)', fontSize: 10, fontFamily: 'var(--font-mono)' }}>
            WASD · Mouse-look · Space/Shift = up/down · Esc = exit
          </div>
        </div>
      )}

      {/* Fly-mode crosshair */}
      {flyMode && flyLocked && (
        <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <svg width="20" height="20" viewBox="0 0 20 20">
            <circle cx="10" cy="10" r="1.5" fill="#4ecdc4" opacity="0.8" />
            <line x1="10" y1="2" x2="10" y2="7"  stroke="#4ecdc4" strokeWidth="1" opacity="0.5" />
            <line x1="10" y1="13" x2="10" y2="18" stroke="#4ecdc4" strokeWidth="1" opacity="0.5" />
            <line x1="2"  y1="10" x2="7"  y2="10" stroke="#4ecdc4" strokeWidth="1" opacity="0.5" />
            <line x1="13" y1="10" x2="18" y2="10" stroke="#4ecdc4" strokeWidth="1" opacity="0.5" />
          </svg>
          <div style={{ position: 'absolute', bottom: 48, left: '50%', transform: 'translateX(-50%)', color: 'rgba(78,205,196,0.35)', fontSize: 9, fontFamily: 'var(--font-mono)', letterSpacing: '0.12em' }}>
            Esc to exit · WASD move · Space/Shift up/down
          </div>
        </div>
      )}

      {/* Loading */}
      {loading && !loadError && (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'rgba(5,5,8,0.75)', flexDirection: 'column', gap: 12,
        }}>
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none"
            stroke="#4ecdc4" strokeWidth="1.5"
            style={{ animation: 'spin 1.2s linear infinite' }}>
            <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
          </svg>
          <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', letterSpacing: '0.12em', color: '#4ecdc4', textTransform: 'uppercase' }}>
            Parsing mesh structure...
          </span>
        </div>
      )}

      {/* Error */}
      {loadError && (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'rgba(5,5,8,0.85)', flexDirection: 'column', gap: 12,
        }}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#ff4f4f" strokeWidth="1.5">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: '#ff4f4f' }}>
            {loadError}
          </span>
        </div>
      )}

      {/* Raycast inspect card */}
      {inspectCard && matchedSurvivor && (
        <div style={{
          position: 'absolute',
          left: Math.min(inspectCard.screenX + 12, window.innerWidth - 200),
          top:  Math.min(inspectCard.screenY - 40, window.innerHeight - 120),
          background: 'rgba(8,12,18,0.92)',
          border: '1px solid rgba(78,205,196,0.3)',
          borderRadius: 8, padding: '10px 14px',
          fontFamily: 'var(--font-mono)', fontSize: 10,
          color: 'var(--text-dim)',
          backdropFilter: 'blur(12px)',
          pointerEvents: 'auto',
          zIndex: 20, minWidth: 160,
        }}>
          <div style={{ color: '#00e676', fontWeight: 700, marginBottom: 6, letterSpacing: '0.08em' }}>
            ✓ Best Survivor
          </div>
          <div style={{ marginBottom: 3 }}>
            Height: <span style={{ color: '#4ecdc4' }}>{matchedSurvivor.height_value.toFixed(1)} {(matchedSurvivor.height_unit || 'REL').slice(0,4)}</span>
          </div>
          <div style={{ marginBottom: 3 }}>
            EGSS: <span style={{ color: '#4ecdc4' }}>{matchedSurvivor.overall_score.toFixed(4)}</span>
          </div>
          <div style={{ marginBottom: 6 }}>
            ID: <span style={{ color: 'var(--text-muted)' }}>{matchedSurvivor.candidate_id}</span>
          </div>
          <button
            onClick={() => setInspectCard(null)}
            style={{ fontSize: 9, color: 'rgba(255,255,255,0.3)', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
          >
            × dismiss
          </button>
        </div>
      )}

      {/* Controls HUD */}
      {!loading && !loadError && (
        <div style={{
          position: 'absolute', bottom: 18, left: '50%', transform: 'translateX(-50%)',
          display: 'flex', alignItems: 'center', gap: 2,
          background: 'rgba(8,9,14,0.82)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(78,205,196,0.15)',
          borderRadius: 24, padding: '5px 10px',
          pointerEvents: 'auto',
        }}>
          <button onClick={resetCamera} style={ctrlBtn}>
            ⌖ Reset
          </button>
          <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)', margin: '0 2px' }} />
          <button
            onClick={() => !flyMode && setAutoRot(r => !r)}
            style={{
              ...ctrlBtn,
              background: (autoRot && !flyMode) ? 'rgba(78,205,196,0.15)' : 'transparent',
              color:       (autoRot && !flyMode) ? '#4ecdc4' : 'rgba(255,255,255,0.35)',
              border:      (autoRot && !flyMode) ? '1px solid rgba(78,205,196,0.3)' : '1px solid transparent',
              opacity:     flyMode ? 0.3 : 1,
            }}
          >
            {(autoRot && !flyMode) ? '⏸ Stop' : '▶ Orbit'}
          </button>
          <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)', margin: '0 2px' }} />
          <button
            onClick={toggleFlyMode}
            style={{
              ...ctrlBtn,
              background: flyMode ? 'rgba(124,111,247,0.2)' : 'transparent',
              color:       flyMode ? '#7c6ff7' : 'rgba(255,255,255,0.35)',
              border:      flyMode ? '1px solid rgba(124,111,247,0.4)' : '1px solid transparent',
            }}
          >
            {flyMode ? '🎮 Exit Fly' : '🚀 Fly'}
          </button>
          <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)', margin: '0 2px' }} />
          <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.2)', padding: '0 6px', fontFamily: 'var(--font-mono)', letterSpacing: '0.06em' }}>
            {flyMode ? 'WASD · mouse-look' : 'Drag · Scroll · Right-pan'}
          </span>
        </div>
      )}

      {/* Mesh info watermark */}
      <div style={{
        position: 'absolute', top: 10, left: '50%', transform: 'translateX(-50%)',
        display: 'flex', gap: 16, alignItems: 'center',
        pointerEvents: 'none',
      }}>
        <span style={{
          fontSize: 9, color: 'rgba(78,205,196,0.3)',
          fontFamily: 'var(--font-mono)', letterSpacing: '0.12em', textTransform: 'uppercase',
        }}>
          AERIS-3D · Vertex-colored terrain · Relative scale
          {meshInfo ? ` · ${(meshInfo.verts / 1000).toFixed(0)}k verts` : ''}
        </span>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

const ctrlBtn = {
  background: 'transparent',
  border: '1px solid transparent',
  cursor: 'pointer',
  color: 'rgba(255,255,255,0.35)',
  padding: '4px 10px',
  borderRadius: 14,
  fontSize: 10,
  fontFamily: 'var(--font-mono)',
  letterSpacing: '0.06em',
  transition: 'all 0.15s',
  outline: 'none',
};
