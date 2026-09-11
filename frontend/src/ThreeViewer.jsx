import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/**
 * AERIS-3D Three.js Viewer
 *
 * Loads a GLB terrain mesh with vertex colors (source image RGB projected onto
 * terrain elevation). Uses MeshBasicMaterial for vertex colors so they render
 * at full brightness without lighting dependency.
 *
 * Controls: Orbit · Pan · Zoom · Auto-rotate flythrough · Camera reset
 */
export default function ThreeViewer({ glbUrl }) {
  const mountRef    = useRef(null);
  const rendererRef = useRef(null);
  const sceneRef    = useRef(null);
  const cameraRef   = useRef(null);
  const controlsRef = useRef(null);
  const rafRef      = useRef(null);
  const autoRotRef  = useRef(false);

  const [loading, setLoading] = useState(true);
  const [autoRot, setAutoRot] = useState(false);
  const [meshInfo, setMeshInfo] = useState(null);

  useEffect(() => {
    if (!glbUrl || !mountRef.current) return;

    const container = mountRef.current;
    const W = container.clientWidth  || window.innerWidth;
    const H = container.clientHeight || window.innerHeight;

    // ── Scene ──────────────────────────────────────────────────
    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#050508');
    sceneRef.current = scene;

    // Very subtle fog — only push distant edges, not mid-range
    scene.fog = new THREE.FogExp2('#050508', 0.04);

    // ── Camera ─────────────────────────────────────────────────
    const camera = new THREE.PerspectiveCamera(50, W / H, 0.001, 500);
    camera.position.set(0, 2.2, 2.8);
    cameraRef.current = camera;

    // ── Renderer ───────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    rendererRef.current = renderer;

    while (container.firstChild) container.removeChild(container.firstChild);
    container.appendChild(renderer.domElement);

    // ── Controls ───────────────────────────────────────────────
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping   = true;
    controls.dampingFactor   = 0.07;
    controls.minDistance     = 0.3;
    controls.maxDistance     = 15;
    controls.maxPolarAngle   = Math.PI / 1.7;
    controls.target.set(0, 0, 0);
    controls.update();
    controlsRef.current = controls;

    // ── Grid ───────────────────────────────────────────────────
    const gridHelper = new THREE.GridHelper(5, 24, 0x4ecdc415, 0x4ecdc410);
    gridHelper.position.y = -0.005;
    scene.add(gridHelper);

    // ── Load GLB ───────────────────────────────────────────────
    const loader = new GLTFLoader();
    loader.load(
      glbUrl,
      (gltf) => {
        const model = gltf.scene;
        let totalVerts = 0;
        let totalFaces = 0;

        model.traverse(node => {
          if (!node.isMesh) return;

          const geo = node.geometry;
          totalVerts += geo.attributes.position?.count || 0;
          totalFaces += (geo.index ? geo.index.count : geo.attributes.position?.count) / 3 | 0;

          if (geo.hasAttribute('color')) {
            // Use MeshBasicMaterial for full-brightness vertex colors
            // (no lighting dependency — colors show exactly as they are in the image)
            node.material = new THREE.MeshBasicMaterial({
              vertexColors: true,
              side: THREE.DoubleSide,
            });
          } else {
            // Fallback: lit material if no vertex colors
            node.material = new THREE.MeshStandardMaterial({
              color: 0x4ecdc4,
              roughness: 0.8,
              metalness: 0.0,
              side: THREE.DoubleSide,
            });
          }
        });

        // Center and scale
        const box    = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size   = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale  = 2.0 / Math.max(maxDim, 0.01);

        model.scale.setScalar(scale);
        model.position.copy(center.multiplyScalar(-scale));

        // DSM: trimesh exports Z as elevation; rotate to Three.js Y-up
        model.rotation.x = -Math.PI / 2;

        // Reposition grid below mesh
        const newBox = new THREE.Box3().setFromObject(model);
        gridHelper.position.y = newBox.min.y - 0.01;

        scene.add(model);
        setMeshInfo({ verts: totalVerts, faces: totalFaces });
        setLoading(false);
      },
      undefined,
      (err) => {
        console.error('GLB load error:', err);
        setLoading(false);
      }
    );

    // ── Animate ────────────────────────────────────────────────
    const animate = () => {
      rafRef.current = requestAnimationFrame(animate);
      if (controlsRef.current) {
        controlsRef.current.autoRotate      = autoRotRef.current;
        controlsRef.current.autoRotateSpeed = 0.7;
        controlsRef.current.update();
      }
      renderer.render(scene, camera);
    };
    animate();

    // ── Resize ─────────────────────────────────────────────────
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
      cancelAnimationFrame(rafRef.current);
      controls.dispose();
      renderer.dispose();
    };
  }, [glbUrl]);

  useEffect(() => { autoRotRef.current = autoRot; }, [autoRot]);

  const resetCamera = () => {
    if (!cameraRef.current || !controlsRef.current) return;
    cameraRef.current.position.set(0, 2.2, 2.8);
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
  };

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Canvas */}
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />

      {/* Loading */}
      {loading && (
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
          <span style={{ color: '#4ecdc4', fontSize: 11, letterSpacing: '0.18em', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
            Loading terrain mesh…
          </span>
        </div>
      )}

      {/* Controls HUD */}
      {!loading && (
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
            onClick={() => setAutoRot(r => !r)}
            style={{
              ...ctrlBtn,
              background: autoRot ? 'rgba(78,205,196,0.15)' : 'transparent',
              color: autoRot ? '#4ecdc4' : 'rgba(255,255,255,0.35)',
              border: autoRot ? '1px solid rgba(78,205,196,0.3)' : '1px solid transparent',
            }}
          >
            {autoRot ? '⏸ Stop' : '▶ Flythrough'}
          </button>
          <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.08)', margin: '0 2px' }} />
          <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.2)', padding: '0 6px', fontFamily: 'var(--font-mono)', letterSpacing: '0.06em' }}>
            Drag · Scroll · Right-drag pan
          </span>
        </div>
      )}

      {/* Mesh info + watermark */}
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
          {meshInfo ? ` · ${(meshInfo.verts/1000).toFixed(0)}k verts` : ''}
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
