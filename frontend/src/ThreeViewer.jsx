import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export default function ThreeViewer({ glbUrl }) {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!glbUrl || !mountRef.current) return;

    // 1. Setup Scene, Camera, Renderer
    const width = mountRef.current.clientWidth;
    const height = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0a0a0f'); // match var(--bg-dark)

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    // Position camera diagonally above
    camera.position.set(2, 2, 2); 

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
    // Optional: enhance color output
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    
    // Clear previous canvas if re-rendering
    while (mountRef.current.firstChild) {
      mountRef.current.removeChild(mountRef.current.firstChild);
    }
    mountRef.current.appendChild(renderer.domElement);

    // 2. Add Controls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // 3. Add Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 10, 5);
    scene.add(dirLight);

    // 4. Load GLB
    const loader = new GLTFLoader();
    
    // Show a loading indicator in the parent component via state ideally, 
    // but here we just load directly.
    loader.load(
      glbUrl,
      (gltf) => {
        const model = gltf.scene;
        
        // Center and scale the model
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        
        // Scale to roughly fit in a 2x2x2 box
        const scale = 2.0 / maxDim;
        model.scale.set(scale, scale, scale);
        
        // Center the model at origin
        model.position.sub(center.multiplyScalar(scale));
        
        // AERIS-3D DSMs are often oriented such that Z is up or Y is up depending on export
        // The trimesh export keeps Z as the elevation. 
        // Three.js uses Y as up. Let's rotate it so Z (elevation) becomes Y (up).
        model.rotation.x = -Math.PI / 2;

        scene.add(model);
      },
      undefined,
      (error) => {
        console.error('An error happened loading the GLB:', error);
      }
    );

    // 5. Animation Loop
    let animationFrameId;
    const animate = () => {
      animationFrameId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    // 6. Handle Resize
    const handleResize = () => {
      if (!mountRef.current) return;
      const w = mountRef.current.clientWidth;
      const h = mountRef.current.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationFrameId);
      controls.dispose();
      renderer.dispose();
    };
  }, [glbUrl]);

  return (
    <div 
      ref={mountRef} 
      style={{ width: '100%', height: '100%', position: 'absolute', top: 0, left: 0, zIndex: 0 }} 
    />
  );
}
