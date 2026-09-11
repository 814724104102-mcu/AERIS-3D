import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/**
 * AERIS-3D Three.js Viewer
 *
 * Loads a GLB terrain mesh and renders it with:
 *  - Vertex colors (source image mapped as texture on terrain)
 *  - Orbit + pan + zoom controls
 *  - Auto-rotate flythrough toggle
 *  - Ambient + directional lighting
 *  - Camera reset
 *  - Responsive resize
 *
 * The terrain mesh is exported from mesh_builder.py with vertex colors
 * derived from the RGB input image. Three.js uses vertexColors rendering.
 */
export default function ThreeViewer({ glbUrl }) {
  const mountRef   = useRef(null);
  const sceneRef   = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef  = useRef(null);
  const controlsRef = useRef(null);
  const rafRef     = useRef(null);
  const autoRotRef = useRef(false);

  const [loading, setLoading]   = useState(true);
  const [autoRot, setAutoRot]   = useState(false);

  useEffect(() => {
    if (!glbUrl || !mountRef.current) return;

    const container = mountRef.current;
    const W = container.clientWidth  || window.innerWidth;
    const H = container.clientHeight || window.innerHeight;

    // ── Scene ─────────────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#050508');

    // Subtle fog for depth feel
    scene.fog = new THREE.FogExp2('#050508', 0.08);
    sceneRef.current = scene;

    // ── Camera ────────────────────────────────────────────────
    const camera = new THREE.PerspectiveCamera(50, W / H, 0.01, 200);
    camera.position.set(0, 2.8, 3.2);
    cameraRef.current = camera;

    // ── Renderer ──────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    while (container.firstChild) container.removeChild(container.firstChild);
    container.appendChild(renderer.domElement);

    // ── Controls ──────────────────────────────────────────────
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance   = 0.5;
    controls.maxDistance   = 20;
    controls.maxPolarAngle = Math.PI / 1.8;
    controls.target.set(0, 0, 0);
    controls.update();
    controlsRef.current = controls;

    // ── Lighting ──────────────────────────────────────────────
    const ambient = new THREE.AmbientLight(0xffffff, 0.55);
    scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xfff8f0, 1.0);
    sun.position.set(4, 8, 5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(1024, 1024);
    scene.add(sun);

    const fill = new THREE.DirectionalLight(0xb0e0e6, 0.3);
    fill.position.set(-4, 2, -5);
    scene.add(fill);

    // ── Grid helper ───────────────────────────────────────────
    const grid = new THREE.GridHelper(6, 20, 0x4ecdc410, 0x4ecdc408);
    grid.position.y = -0.01;
    scene.add(grid);

    // ── Load GLB ──────────────────────────────────────────────
    const loader = new GLTFLoader();
    loader.load(
      glbUrl,
      (gltf) => {
        const model = gltf.scene;

        // Enable vertex colors on all meshes
        model.traverse(node => {
          if (node.isMesh) {
            node.castShadow    = true;
            node.receiveShadow = true;
            // Preserve vertex colors
            if (node.geometry.hasAttribute('color')) {
              node.material.vertexColors = true;
              node.material.needsUpdate  = true;
            }
          }
        });

        // Center and scale to fit
        const box    = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size   = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale  = 2.2 / Math.max(maxDim, 0.01);

        model.scale.setScalar(scale);
        model.position.sub(center.multiplyScalar(scale));

        // DSM: Z is elevation (trimesh default). Rotate so Z→Y for Three.js
        model.rotation.x = -Math.PI / 2;

        scene.add(model);
        setLoading(false);
      },
      undefined,
      (err) => {
        console.error('GLB load error:', err);
        setLoading(false);
      }
    );

    // ── Animation loop ────────────────────────────────────────
    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);
      if (autoRotRef.current && controlsRef.current) {
        controlsRef.current.autoRotate      = true;
        controlsRef.current.autoRotateSpeed = 0.8;
      } else if (controlsRef.current) {
        controlsRef.current.autoRotate = false;
      }
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // ── Resize ────────────────────────────────────────────────
    const onResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    return () => {
      window.removeEventListener('resize', onResize);
      cancelAnimationFrame(rafRef.current);
      controls.dispose();
      renderer.dispose();
    };
  }, [glbUrl]);

  // Keep autoRotRef in sync with state
  useEffect(() => {
    autoRotRef.current = autoRot;
  }, [autoRot]);

  const resetCamera = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.set(0, 2.8, 3.2);
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
  };

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Canvas mount */}
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {/* Loading overlay */}
      {loading && (
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          background: 'rgba(5,5,8,0.7)',
          color: 'var(--accent)', fontSize: 12, letterSpacing: '0.15em',
          textTransform: 'uppercase', fontWeight: 600,
        }}>
          <div style={{ textAlign: 'center' }}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
              style={{ display: 'block', margin: '0 auto 10px', animation: 'spin 1.2s linear infinite' }}>
              <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
            </svg>
            Loading terrain mesh…
          </div>
        </div>
      )}

      {/* 3D Controls HUD */}
      {!loading && (
        <div style={{
          position: 'absolute', bottom: 20, left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex', gap: 8,
          background: 'rgba(10,10,15,0.75)',
          backdropFilter: 'blur(12px)',
          border: '1px solid var(--border)',
          borderRadius: 20, padding: '6px 12px',
          pointerEvents: 'auto',
        }}>
          <button
            title="Reset camera"
            onClick={resetCamera}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'var(--text-dim)', padding: '4px 8px', borderRadius: 12,
              fontSize: 10, fontFamily: 'var(--text-mono)', letterSpacing: '0.06em',
              transition: 'color 0.15s',
            }}
            onMouseOver={e => e.currentTarget.style.color = 'var(--accent)'}
            onMouseOut={e => e.currentTarget.style.color = 'var(--text-dim)'}
          >
            ⌖ Reset
          </button>
          <div style={{ width: 1, background: 'var(--border)', margin: '2px 0' }} />
          <button
            title="Toggle flythrough"
            onClick={() => setAutoRot(r => !r)}
            style={{
              background: autoRot ? 'var(--accent-dim)' : 'none',
              border: '1px solid ' + (autoRot ? 'var(--accent-border)' : 'transparent'),
              cursor: 'pointer',
              color: autoRot ? 'var(--accent)' : 'var(--text-dim)',
              padding: '4px 10px', borderRadius: 12,
              fontSize: 10, fontFamily: 'var(--text-mono)', letterSpacing: '0.06em',
              transition: 'all 0.15s',
            }}
          >
            {autoRot ? '⏸ Stop' : '▶ Flythrough'}
          </button>
          <div style={{ width: 1, background: 'var(--border)', margin: '2px 0' }} />
          <span style={{ fontSize: 9, color: 'var(--text-muted)', padding: '4px 6px', alignSelf: 'center', fontFamily: 'var(--text-mono)' }}>
            Drag · Scroll · Right-drag pan
          </span>
        </div>
      )}

      {/* Scale watermark */}
      <div style={{
        position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)',
        fontSize: 9, color: 'rgba(78,205,196,0.25)', fontFamily: 'var(--text-mono)',
        letterSpacing: '0.12em', textTransform: 'uppercase', pointerEvents: 'none',
      }}>
        AERIS-3D · Vertex-colored terrain · Relative scale
      </div>
    </div>
  );
}
