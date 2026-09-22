import re

with open(r'n:\AERIS-3D-main\backend\viewer.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 1. Add PointerLockControls script
html = html.replace(
    '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>',
    '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>\n    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/PointerLockControls.js"></script>'
)

# 2. Add UI elements to viewer-overlay
overlay_injection = """
                <button class="btn-ctrl" id="btn-pointerlock" onclick="startPointerLock()">Fly Mode (PointerLock)</button>
                <button class="btn-ctrl" id="btn-slope" onclick="toggleSlopeTool()">Slope Tool</button>
                <button class="btn-ctrl" id="btn-overlay" onclick="toggleReferenceOverlay()">Ref Overlay</button>
                <div id="height-readout" style="color: #34d399; font-family: monospace; display: flex; align-items: center; margin-left: 10px;">Height: --</div>
                <div id="slope-readout" style="color: #fb923c; font-family: monospace; display: flex; align-items: center; margin-left: 10px;">Slope: --</div>
"""
html = html.replace('<button class="btn-ctrl" onclick="resetCamera()">Reset Camera</button>',
                   '<button class="btn-ctrl" onclick="resetCamera()">Reset Camera</button>' + overlay_injection)

# 3. Add global variables
js_globals = """
        let pointerLockControls;
        let isPointerLocked = false;
        let raycaster = new THREE.Raycaster();
        let mouse = new THREE.Vector2(0, 0); // Center of screen for pointer lock, or mouse pos
        let slopeMode = false;
        let slopePoints = [];
        let slopeMarkers = [];
        let referenceMode = false;
        let originalMaterial = null;
        let referenceMaterial = new THREE.MeshBasicMaterial({color: 0xff0000, wireframe: true}); // Mock for reference
"""
html = html.replace('let isWireframe = false;', 'let isWireframe = false;\n' + js_globals)

# 4. Initialize PointerLockControls and Event Listeners
init_code = """
            pointerLockControls = new THREE.PointerLockControls(camera, document.body);
            pointerLockControls.addEventListener('lock', () => { isPointerLocked = true; });
            pointerLockControls.addEventListener('unlock', () => { isPointerLocked = false; });
            
            document.addEventListener('click', onDocumentClick, false);
"""
html = html.replace('controls.maxPolarAngle = Math.PI / 2 - 0.05;', 'controls.maxPolarAngle = Math.PI / 2 - 0.05;\n' + init_code)

# 5. Add functions and update animate()
functions_code = """
        function startPointerLock() {
            pointerLockControls.lock();
        }

        function toggleSlopeTool() {
            slopeMode = !slopeMode;
            document.getElementById('btn-slope').classList.toggle('active', slopeMode);
            slopePoints = [];
            slopeMarkers.forEach(m => scene.remove(m));
            slopeMarkers = [];
            document.getElementById('slope-readout').textContent = 'Slope: --';
        }

        function toggleReferenceOverlay() {
            referenceMode = !referenceMode;
            document.getElementById('btn-overlay').classList.toggle('active', referenceMode);
            if (currentMesh) {
                currentMesh.traverse((child) => {
                    if (child.isMesh) {
                        if (referenceMode) {
                            if (!originalMaterial) originalMaterial = child.material;
                            child.material = referenceMaterial;
                        } else {
                            if (originalMaterial) child.material = originalMaterial;
                        }
                    }
                });
            }
        }

        function onDocumentClick(event) {
            if (!slopeMode || !currentMesh) return;
            
            // If pointer locked, use center. Else use mouse coordinates
            if (!isPointerLocked) {
                const rect = renderer.domElement.getBoundingClientRect();
                mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            } else {
                mouse.x = 0; mouse.y = 0;
            }

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObject(currentMesh, true);
            
            if (intersects.length > 0) {
                const pt = intersects[0].point;
                slopePoints.push(pt);
                
                const geo = new THREE.SphereGeometry(1, 16, 16);
                const mat = new THREE.MeshBasicMaterial({color: 0xfb923c});
                const sphere = new THREE.Mesh(geo, mat);
                sphere.position.copy(pt);
                scene.add(sphere);
                slopeMarkers.push(sphere);

                if (slopePoints.length === 2) {
                    const p1 = slopePoints[0];
                    const p2 = slopePoints[1];
                    const dy = Math.abs(p2.y - p1.y);
                    const dxz = Math.hypot(p2.x - p1.x, p2.z - p1.z);
                    const angleRad = Math.atan2(dy, dxz);
                    const angleDeg = (angleRad * 180 / Math.PI).toFixed(1);
                    const grade = ((dy / dxz) * 100).toFixed(1);
                    document.getElementById('slope-readout').textContent = `Slope: ${angleDeg}° (${grade}%)`;
                    
                    // reset for next pair
                    slopePoints = [];
                    setTimeout(() => {
                        slopeMarkers.forEach(m => scene.remove(m));
                        slopeMarkers = [];
                    }, 3000);
                }
            }
        }

        // We will override animate to add our features
"""

html = html.replace('function toggleFlythrough() {', functions_code + 'function toggleFlythrough() {')

# Modify animate to do raycasting and clamping
animate_override = """
        function animate() {
            requestAnimationFrame(animate);

            if (isFlythrough) {
                angle += 0.005;
                const radius = 90;
                camera.position.x = Math.sin(angle) * radius;
                camera.position.z = Math.cos(angle) * radius;
                camera.position.y = 45 + Math.sin(angle * 2) * 10;
                controls.target.set(0, 5, 0);
            }

            // Height readout
            if (currentMesh) {
                raycaster.setFromCamera(new THREE.Vector2(0, 0), camera);
                const intersects = raycaster.intersectObject(currentMesh, true);
                if (intersects.length > 0) {
                    const h = intersects[0].point.y;
                    document.getElementById('height-readout').textContent = `Height: ${h.toFixed(2)}m`;
                } else {
                    document.getElementById('height-readout').textContent = `Height: --`;
                }
                
                // Collision / Clamping
                if (isPointerLocked || isFlythrough || true) {
                    // Raycast down from camera X,Z to find terrain height
                    const downRay = new THREE.Raycaster(new THREE.Vector3(camera.position.x, 1000, camera.position.z), new THREE.Vector3(0, -1, 0));
                    const downHits = downRay.intersectObject(currentMesh, true);
                    if (downHits.length > 0) {
                        const groundY = downHits[0].point.y;
                        if (camera.position.y < groundY + 2) {
                            camera.position.y = groundY + 2;
                        }
                    }
                }
            }

            if (!isPointerLocked) controls.update();
            renderer.render(scene, camera);
        }
"""
# Replace the original animate block
html = re.sub(r'function animate\(\) \{[\s\S]*?renderer\.render\(scene, camera\);\s*\}', animate_override, html)

with open(r'n:\AERIS-3D-main\backend\viewer.html', 'w', encoding='utf-8') as f:
    f.write(html)
