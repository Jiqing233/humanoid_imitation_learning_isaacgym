#!/usr/bin/env python3
"""
Generate an interactive HTML visualization of Isaac and Unity skeletons.

Uses Three.js for 3D rendering — no matplotlib dependency.
Opens in any browser for interactive rotation/zoom.

Usage:
    conda run -n isaac_gym python visualize_skeletons.py
    # Then open exports/skeleton_viewer.html in a browser
"""
import json
import os
import sys
import webbrowser
import numpy as np


# ============================================================
# Quaternion math (XYZW)
# ============================================================

def qn(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0, 0, 0, 1.0])

def qm(q1, q2):
    x1, y1, z1, w1 = q1; x2, y2, z2, w2 = q2
    return qn(np.array([
        w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2]))

def qr(q, v):
    """Rotate vector v by unit quaternion q. Preserves magnitude."""
    u = q[:3]; w = q[3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


# ============================================================
# I/O & FK
# ============================================================

def v3(d): return np.array([d["x"], d["y"], d["z"]], dtype=np.float64)
def q4(d): return qn(np.array([d["x"], d["y"], d["z"], d["w"]], dtype=np.float64))

def fk(parents, lpos, lrot):
    n = len(parents)
    wp, wr = [None]*n, [None]*n
    for i in range(n):
        p = parents[i]
        if p == -1:
            wp[i], wr[i] = lpos[i].copy(), lrot[i].copy()
        else:
            wr[i] = qm(wr[p], lrot[i])
            wp[i] = wp[p] + qr(wr[p], lpos[i])
    return wp, wr

def isaac_to_unity_pos(p):
    return np.array([-p[1], p[2], p[0]])


# ============================================================
# Load skeletons and compute world positions
# ============================================================

def load_isaac(path):
    data = json.load(open(path))
    lp = [v3(d) for d in data["source_bind_local_pos"]]
    lr = [q4(d) for d in data["source_bind_local_rot"]]
    wp, _ = fk(data["parent_indices"], lp, lr)
    # Convert to Unity coords (Y-up) for unified visualization
    wp_unity = [isaac_to_unity_pos(p) for p in wp]
    return data["bone_names"], data["parent_indices"], wp_unity


def load_unity(path, frame_rots=None):
    data = json.load(open(path))
    lp = [v3(d) for d in data["target_bind_local_pos"]]
    lr = [q4(d) for d in data["target_bind_local_rot"]]
    if frame_rots is not None:
        lr = [q4(d) for d in frame_rots]
    wp, _ = fk(data["parent_indices"], lp, lr)
    return data["bone_names"], data["parent_indices"], wp


def load_retargeted(path):
    data = json.load(open(path))
    lp = [v3(d) for d in data["target_bind_local_pos"]]
    poses = []
    for frame in data["frames"]:
        lr = [q4(d) for d in frame["target_local_rot"]]
        wp, _ = fk(data["parent_indices"], lp, lr)
        label = frame.get("label", f"frame_{len(poses)}")
        poses.append((label, wp))
    return data["bone_names"], data["parent_indices"], poses


# ============================================================
# Build skeleton data for Three.js (Y-up, consistent with Unity)
# ============================================================

BODY_COLORS = {
    "pelvis": "#888", "torso": "#4a4", "head": "#a4a",
    "right_upper_arm": "#e55", "right_lower_arm": "#e55", "right_hand": "#e55",
    "left_upper_arm": "#55e", "left_lower_arm": "#55e", "left_hand": "#55e",
    "right_thigh": "#e84", "right_shin": "#e84", "right_foot": "#e84",
    "left_thigh": "#48e", "left_shin": "#48e", "left_foot": "#48e",
    "Hips": "#888",
    "Spine 1": "#4a4", "Spine 2": "#4a4", "Spine 3": "#4a4",
    "Neck": "#a4a", "Head": "#a4a",
    "Right Shoulder": "#e55", "Right Arm": "#e55", "Right Forearm": "#e55", "Right Hand": "#e55",
    "Left Shoulder": "#55e", "Left Arm": "#55e", "Left Forearm": "#55e", "Left Hand": "#55e",
    "Right Thigh": "#e84", "Right Leg": "#e84", "Right Foot": "#e84",
    "Left Thigh": "#48e", "Left Leg": "#48e", "Left Foot": "#48e",
}


def skeleton_to_joints(names, parents, world_pos, x_offset=0.0):
    """Convert skeleton to list of joint dicts for JSON serialization."""
    joints = []
    for i, name in enumerate(names):
        p = world_pos[i]
        color = BODY_COLORS.get(name, "#aaa")
        joints.append({
            "name": name,
            "parent": parents[i],
            "pos": [float(p[0] + x_offset), float(p[1]), float(p[2])],
            "color": color,
        })
    return joints


# ============================================================
# HTML template
# ============================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Skeleton Comparison</title>
<style>
  body { margin:0; overflow:hidden; background:#1a1a2e; font-family:monospace; }
  #info { position:absolute; top:8px; left:8px; color:#ccc; font-size:12px; z-index:10;
          background:rgba(0,0,0,0.6); padding:8px 12px; border-radius:4px; }
  #info h3 { margin:0 0 4px 0; color:#fff; }
  #controls { position:absolute; bottom:8px; left:8px; color:#ccc; font-size:11px; z-index:10;
              background:rgba(0,0,0,0.6); padding:6px 10px; border-radius:4px; }
  #pose-select { position:absolute; top:8px; right:8px; z-index:10;
                 background:rgba(0,0,0,0.6); padding:8px; border-radius:4px; }
  #pose-select button { margin:2px; padding:4px 8px; cursor:pointer;
                        background:#333; color:#ccc; border:1px solid #555; border-radius:3px; }
  #pose-select button.active { background:#558; color:#fff; border-color:#88a; }
</style>
</head>
<body>
<div id="info">
  <h3>Skeleton Comparison</h3>
  <div id="hover-info">Hover over joints to see names</div>
</div>
<div id="controls">
  Drag to rotate | Scroll to zoom | Right-drag to pan<br>
  <span style="color:#e55">&#9679;</span> Right arm &nbsp;
  <span style="color:#55e">&#9679;</span> Left arm &nbsp;
  <span style="color:#e84">&#9679;</span> Right leg &nbsp;
  <span style="color:#48e">&#9679;</span> Left leg &nbsp;
  <span style="color:#4a4">&#9679;</span> Spine &nbsp;
  <span style="color:#a4a">&#9679;</span> Head
</div>
<div id="pose-select"></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
// ---- Skeleton data (injected by Python) ----
const SKELETONS = %%SKELETONS%%;

// ---- Scene setup ----
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth/window.innerHeight, 0.001, 100);
camera.position.set(0, 0.8, 2.5);

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.6, 0);
controls.update();

// Grid
const grid = new THREE.GridHelper(4, 20, 0x333355, 0x222244);
scene.add(grid);

// Lights
scene.add(new THREE.AmbientLight(0x404040));
const dl = new THREE.DirectionalLight(0xffffff, 0.8);
dl.position.set(2, 4, 3);
scene.add(dl);

// ---- Build skeleton meshes ----
const skeletonGroups = {};
const allJointMeshes = [];

function buildSkeleton(key, data) {
  const group = new THREE.Group();
  const joints = data.joints;

  for (let i = 0; i < joints.length; i++) {
    const j = joints[i];
    // Joint sphere
    const geo = new THREE.SphereGeometry(data.jointRadius || 0.015, 8, 8);
    const mat = new THREE.MeshPhongMaterial({color: j.color});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(j.pos[0], j.pos[1], j.pos[2]);
    mesh.userData = {name: j.name, skeleton: data.title};
    group.add(mesh);
    allJointMeshes.push(mesh);

    // Bone line to parent
    if (j.parent >= 0) {
      const pj = joints[j.parent];
      const pts = [
        new THREE.Vector3(pj.pos[0], pj.pos[1], pj.pos[2]),
        new THREE.Vector3(j.pos[0], j.pos[1], j.pos[2])
      ];
      const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const lineMat = new THREE.LineBasicMaterial({color: j.color, linewidth: 2});
      group.add(new THREE.Line(lineGeo, lineMat));
    }

    // Label (sprite)
    if (data.showLabels !== false) {
      const canvas = document.createElement('canvas');
      canvas.width = 256; canvas.height = 40;
      const ctx = canvas.getContext('2d');
      ctx.font = '20px monospace';
      ctx.fillStyle = j.color;
      ctx.fillText(j.name, 2, 28);
      const tex = new THREE.CanvasTexture(canvas);
      const spriteMat = new THREE.SpriteMaterial({map:tex, transparent:true});
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.set(j.pos[0]+0.02, j.pos[1]+0.02, j.pos[2]);
      sprite.scale.set(0.15, 0.025, 1);
      group.add(sprite);
    }
  }

  // Title label above skeleton
  const titleCanvas = document.createElement('canvas');
  titleCanvas.width = 512; titleCanvas.height = 48;
  const tctx = titleCanvas.getContext('2d');
  tctx.font = 'bold 28px monospace';
  tctx.fillStyle = '#ffffff';
  tctx.textAlign = 'center';
  tctx.fillText(data.title, 256, 34);
  const ttex = new THREE.CanvasTexture(titleCanvas);
  const tmat = new THREE.SpriteMaterial({map:ttex, transparent:true});
  const tsprite = new THREE.Sprite(tmat);
  const maxY = Math.max(...joints.map(j=>j.pos[1]));
  tsprite.position.set(joints[0].pos[0], maxY + 0.12, 0);
  tsprite.scale.set(0.6, 0.06, 1);
  group.add(tsprite);

  scene.add(group);
  skeletonGroups[key] = group;
  return group;
}

// Build all skeletons
const keys = Object.keys(SKELETONS);
keys.forEach(k => buildSkeleton(k, SKELETONS[k]));

// ---- Pose selection buttons ----
const poseDiv = document.getElementById('pose-select');
const poseKeys = keys.filter(k => SKELETONS[k].group === 'retarget');
const baseKeys = keys.filter(k => SKELETONS[k].group !== 'retarget');

if (poseKeys.length > 0) {
  // Show first retarget pose, hide rest
  poseKeys.forEach((k, i) => {
    if (i > 0) skeletonGroups[k].visible = false;
    const btn = document.createElement('button');
    btn.textContent = SKELETONS[k].title.replace('Retarget: ','');
    btn.className = i === 0 ? 'active' : '';
    btn.onclick = () => {
      poseKeys.forEach(pk => { skeletonGroups[pk].visible = false; });
      skeletonGroups[k].visible = true;
      poseDiv.querySelectorAll('button').forEach(b => b.className = '');
      btn.className = 'active';
    };
    poseDiv.appendChild(btn);
  });
}

// ---- Hover detection ----
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const hoverInfo = document.getElementById('hover-info');

renderer.domElement.addEventListener('mousemove', e => {
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
});

// ---- Animate ----
function animate() {
  requestAnimationFrame(animate);

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(allJointMeshes);
  if (hits.length > 0) {
    const d = hits[0].object.userData;
    hoverInfo.textContent = `${d.skeleton}: ${d.name}`;
  } else {
    hoverInfo.textContent = 'Hover over joints to see names';
  }

  controls.update();
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
</script>
</body>
</html>"""


# ============================================================
# Main
# ============================================================

def main():
    isaac_path = "exports/isaac_static_skeleton.json"
    unity_path = "unity/target_skeleton_main.json"
    retarget_path = "exports/motion_data/retargeted_test_poses.json"
    output_html = "exports/skeleton_viewer.html"

    has_isaac = os.path.exists(isaac_path)
    has_retarget = os.path.exists(retarget_path)

    if not has_isaac:
        print(f"[WARN] {isaac_path} not found. Run extract_isaac_skeleton.py first.")

    # Load Unity target skeleton
    unity_names, unity_parents, unity_wp = load_unity(unity_path)

    # ---- Normalization ----
    # Both skeletons get scaled so their max bounding-box dimension = 1.5,
    # then shifted so the lowest point sits on Y=0 (Three.js Y-up).
    # The Unity skeleton may have Z-up orientation from its root rotation,
    # so we detect the actual "up" axis by finding the hips-to-head direction.

    def normalize_skeleton(world_pos, target_height=1.5):
        """Scale, re-orient, and center a skeleton for Three.js (Y-up).

        Detects the skeleton's actual up-axis by looking at the largest
        bounding-box extent. If the skeleton is Z-up, applies Rx(-90 deg)
        to rotate it to Y-up.  Then scales to target_height and shifts
        so the lowest point sits at Y=0.
        Returns (normalized_pos, scale).
        """
        pts = np.array(world_pos)
        extents = pts.max(axis=0) - pts.min(axis=0)  # [dx, dy, dz]

        # If Z-extent > Y-extent, the skeleton is Z-up -> rotate to Y-up
        # Rx(-90 deg): (x, y, z) -> (x, z, -y)
        if extents[2] > extents[1] * 1.5:
            pts = np.column_stack([pts[:, 0], pts[:, 2], -pts[:, 1]])
            extents = pts.max(axis=0) - pts.min(axis=0)

        # Scale so max extent = target_height
        max_dim = extents.max()
        scale = target_height / max_dim if max_dim > 1e-8 else 1.0
        pts = pts * scale
        # Shift feet to Y=0
        pts[:, 1] -= pts[:, 1].min()
        return [pts[i] for i in range(len(world_pos))], scale

    skeletons = {}
    x_cursor = 0.0  # horizontal position for side-by-side layout

    # --- Isaac skeleton (A-pose, converted to Unity Y-up) ---
    if has_isaac:
        i_names, i_parents, i_wp = load_isaac(isaac_path)
        i_wp_norm, i_scale = normalize_skeleton(i_wp)

        skeletons["isaac"] = {
            "title": "Isaac A-pose",
            "group": "source",
            "joints": skeleton_to_joints(i_names, i_parents, i_wp_norm, x_offset=0),
            "jointRadius": 0.02,
        }

        print("\nIsaac A-pose world positions (Unity coords, normalized):")
        print(f"  {'Bone':<22s} {'X(right)':>9s} {'Y(up)':>9s} {'Z(fwd)':>9s}")
        for i, n in enumerate(i_names):
            p = i_wp_norm[i]
            print(f"  {n:<22s} {p[0]:9.4f} {p[1]:9.4f} {p[2]:9.4f}")

        x_cursor += 1.2

    # --- Unity target skeleton (A-pose / bind) ---
    unity_wp_norm, u_scale = normalize_skeleton(unity_wp)

    skeletons["unity"] = {
        "title": "Unity Target A-pose",
        "group": "target",
        "joints": skeleton_to_joints(unity_names, unity_parents, unity_wp_norm, x_offset=x_cursor),
        "jointRadius": 0.02,
    }

    print(f"\nUnity Target A-pose world positions (normalized):")
    print(f"  {'Bone':<22s} {'X(right)':>9s} {'Y(up)':>9s} {'Z(fwd)':>9s}")
    for i, n in enumerate(unity_names):
        p = unity_wp_norm[i]
        print(f"  {n:<22s} {p[0]:9.4f} {p[1]:9.4f} {p[2]:9.4f}")

    x_cursor += 1.2

    # --- Retargeted poses ---
    if has_retarget:
        rt_names, rt_parents, rt_poses = load_retargeted(retarget_path)
        for pi, (label, wp) in enumerate(rt_poses):
            rt_wp_norm, _ = normalize_skeleton(wp)
            skeletons[f"retarget_{pi}"] = {
                "title": f"Retarget: {label}",
                "group": "retarget",
                "showLabels": False,
                "joints": skeleton_to_joints(rt_names, rt_parents, rt_wp_norm, x_offset=x_cursor),
                "jointRadius": 0.02,
            }
        print(f"\nLoaded {len(rt_poses)} retargeted poses: {[l for l,_ in rt_poses]}")

    # --- Generate HTML ---
    skeletons_json = json.dumps(skeletons, indent=2)
    html = HTML_TEMPLATE.replace("%%SKELETONS%%", skeletons_json)

    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    with open(output_html, "w") as f:
        f.write(html)

    print(f"\nSaved interactive viewer to {output_html}")
    print("Open in a browser to view.")

    # Try to open in browser
    try:
        webbrowser.open(f"file://{os.path.abspath(output_html)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
