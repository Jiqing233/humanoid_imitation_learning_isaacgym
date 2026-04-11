#!/usr/bin/env python3
"""
Convert Isaac Gym .npy motion to Unity avatar animation.

Reads an Isaac Gym motion .npy file, retargets every frame to the Unity
skeleton using world-space deltas with proper coordinate conversion,
and outputs:
  1. Unity-format JSON animation file
  2. Interactive HTML animation viewer for side-by-side validation

Usage:
    conda run -n isaac_gym python convert_motion.py data/martial_arts/Garren_Knifehand_amp.npy
    conda run -n isaac_gym python convert_motion.py data/martial_arts/Garren_Knifehand_amp.npy --config configs/isaac_to_banana.json
"""
import json
import math
import os
import sys
import webbrowser
import numpy as np

from retarget_skeleton import (
    quat_identity, quat_normalize, quat_mul, quat_rotate,
    isaac_pos_to_unity, isaac_quat_to_unity,
    v3, q4, v3d, q4d, load_json, save_json,
    fk_full, build_bone_mapping, retarget_frame,
)


# ============================================================
# Load .npy motion
# ============================================================

def load_npy_motion(path):
    """Load Isaac Gym .npy motion file (SkeletonMotion format)."""
    data = np.load(path, allow_pickle=True).item()
    skel = data["skeleton_tree"]
    return {
        "bone_names": list(skel["node_names"]),
        "parent_indices": skel["parent_indices"]["arr"].astype(int).tolist(),
        "local_translation": skel["local_translation"]["arr"].astype(np.float64),
        "rotation": data["rotation"]["arr"].astype(np.float64),
        "root_translation": data["root_translation"]["arr"].astype(np.float64),
        "fps": int(data.get("fps", 60)),
        "num_frames": data["rotation"]["arr"].shape[0],
        "num_joints": data["rotation"]["arr"].shape[1],
    }


# ============================================================
# Build source A-pose bind
# ============================================================

def make_isaac_a_pose_bind(bone_names, local_translation):
    """Create A-pose bind rotations for Isaac skeleton (±45° upper arm X)."""
    n = len(bone_names)
    bind_pos = [local_translation[j].copy() for j in range(n)]
    bind_rot = [quat_identity() for _ in range(n)]
    half = math.radians(45.0) / 2.0
    s, c = math.sin(half), math.cos(half)
    idx = {name: i for i, name in enumerate(bone_names)}
    if "right_upper_arm" in idx:
        bind_rot[idx["right_upper_arm"]] = np.array([-s, 0.0, 0.0, c])
    if "left_upper_arm" in idx:
        bind_rot[idx["left_upper_arm"]] = np.array([s, 0.0, 0.0, c])
    return bind_pos, bind_rot


# ============================================================
# Body-part colours (shared with visualize_skeletons.py)
# ============================================================

COLORS_ISAAC = {
    "pelvis": "#888", "torso": "#4a4", "head": "#a4a",
    "right_upper_arm": "#e55", "right_lower_arm": "#e55", "right_hand": "#e55",
    "left_upper_arm": "#55e", "left_lower_arm": "#55e", "left_hand": "#55e",
    "right_thigh": "#e84", "right_shin": "#e84", "right_foot": "#e84",
    "left_thigh": "#48e", "left_shin": "#48e", "left_foot": "#48e",
}
COLORS_UNITY = {
    "Hips": "#888",
    "Spine 1": "#4a4", "Spine 2": "#4a4", "Spine 3": "#4a4",
    "Neck": "#a4a", "Head": "#a4a",
    "Right Shoulder": "#e55", "Right Arm": "#e55", "Right Forearm": "#e55", "Right Hand": "#e55",
    "Left Shoulder": "#55e", "Left Arm": "#55e", "Left Forearm": "#55e", "Left Hand": "#55e",
    "Right Thigh": "#e84", "Right Leg": "#e84", "Right Foot": "#e84",
    "Left Thigh": "#48e", "Left Leg": "#48e", "Left Foot": "#48e",
}


# ============================================================
# Generate animation HTML viewer
# ============================================================

def generate_animation_html(
    out_path,
    src_names, src_parents, src_frames_wp,
    tgt_names, tgt_parents, tgt_frames_wp,
    fps,
):
    """Create an HTML file with a Three.js animation player.

    src_frames_wp / tgt_frames_wp: list-of-lists  [T][J] = np.array(3)
    """
    T = len(src_frames_wp)

    # --- Normalise both skeletons to similar scale ---
    # Find global bounding box across ALL frames for each skeleton
    all_src = np.array([[p for p in frame] for frame in src_frames_wp])  # (T,J,3)
    all_tgt = np.array([[p for p in frame] for frame in tgt_frames_wp])  # (T,J,3)

    def norm_params(pts):
        """Compute (scale, offset) to normalise pts to ~1.5 height, feet at Y=0."""
        flat = pts.reshape(-1, 3)
        extents = flat.max(0) - flat.min(0)
        # If Z-extent > Y-extent, skeleton is Z-up -> swap to Y-up
        swap_yz = extents[2] > extents[1] * 1.5
        if swap_yz:
            flat = np.column_stack([flat[:, 0], flat[:, 2], -flat[:, 1]])
            extents = flat.max(0) - flat.min(0)
        scale = 1.5 / max(extents.max(), 1e-8)
        y_min = flat[:, 1].min()
        return scale, y_min, swap_yz

    src_scale, src_ymin, src_swap = norm_params(all_src)
    tgt_scale, tgt_ymin, tgt_swap = norm_params(all_tgt)

    def apply_norm(pts_3d, scale, ymin, swap, x_off=0.0):
        """Return [x,y,z] list after normalisation."""
        x, y, z = float(pts_3d[0]), float(pts_3d[1]), float(pts_3d[2])
        if swap:
            y, z = z, -y
        return [round((x) * scale + x_off, 5),
                round((y - ymin) * scale, 5),
                round(z * scale, 5)]

    # Build compact frame data:  frames[t] = [[x,y,z], ...]  per joint
    x_off_src = 0.0
    x_off_tgt = 1.3

    src_frame_data = []
    for t in range(T):
        src_frame_data.append(
            [apply_norm(src_frames_wp[t][j], src_scale, src_ymin, src_swap, x_off_src)
             for j in range(len(src_names))])

    tgt_frame_data = []
    for t in range(T):
        tgt_frame_data.append(
            [apply_norm(tgt_frames_wp[t][j], tgt_scale, tgt_ymin, tgt_swap, x_off_tgt)
             for j in range(len(tgt_names))])

    # Build JSON payload
    anim_json = json.dumps({
        "fps": fps,
        "numFrames": T,
        "source": {
            "title": "Isaac Source",
            "names": src_names,
            "parents": src_parents,
            "colors": [COLORS_ISAAC.get(n, "#aaa") for n in src_names],
            "frames": src_frame_data,
        },
        "target": {
            "title": "Unity Retarget",
            "names": tgt_names,
            "parents": tgt_parents,
            "colors": [COLORS_UNITY.get(n, "#aaa") for n in tgt_names],
            "frames": tgt_frame_data,
        },
    }, separators=(',', ':'))  # compact JSON

    html = ANIM_HTML_TEMPLATE.replace("%%ANIM_DATA%%", anim_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)


ANIM_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Motion Retarget Viewer</title>
<style>
  body{margin:0;overflow:hidden;background:#1a1a2e;font-family:monospace;color:#ccc}
  #hud{position:absolute;top:0;left:0;right:0;padding:8px 12px;
       background:rgba(0,0,0,.6);display:flex;align-items:center;gap:12px;z-index:10;flex-wrap:wrap}
  #hud button{padding:4px 12px;cursor:pointer;background:#335;color:#ccc;border:1px solid #558;border-radius:3px}
  #hud button:hover{background:#447}
  #hud button.active{background:#558;color:#fff}
  #slider{flex:1;min-width:200px}
  #info{position:absolute;bottom:8px;left:8px;font-size:11px;background:rgba(0,0,0,.5);padding:4px 8px;border-radius:3px;z-index:10}
  #hover-info{position:absolute;bottom:30px;left:8px;font-size:11px;background:rgba(0,0,0,.5);padding:4px 8px;border-radius:3px;z-index:10}
  .legend{font-size:11px}
</style>
</head>
<body>
<div id="hud">
  <button id="playBtn" onclick="togglePlay()">&#9654; Play</button>
  <input id="slider" type="range" min="0" max="1" value="0" step="1">
  <span id="frameLabel">0 / 0</span>
  <span>Speed:</span>
  <button onclick="setSpeed(0.25)">0.25x</button>
  <button onclick="setSpeed(0.5)">0.5x</button>
  <button class="active" id="speed1" onclick="setSpeed(1)">1x</button>
  <button onclick="setSpeed(2)">2x</button>
  <span class="legend">
    <span style="color:#e55">&#9679;</span>R-arm
    <span style="color:#55e">&#9679;</span>L-arm
    <span style="color:#e84">&#9679;</span>R-leg
    <span style="color:#48e">&#9679;</span>L-leg
    <span style="color:#4a4">&#9679;</span>Spine
  </span>
</div>
<div id="hover-info">Hover joints for names</div>
<div id="info">Drag=rotate | Scroll=zoom | Right-drag=pan | Arrow keys=step frames</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const D = %%ANIM_DATA%%;
const slider = document.getElementById('slider');
const frameLabel = document.getElementById('frameLabel');
const playBtn = document.getElementById('playBtn');
slider.max = D.numFrames - 1;
frameLabel.textContent = `0 / ${D.numFrames-1}`;

let playing = false, speed = 1, accumTime = 0, currentFrame = 0;

function togglePlay() { playing = !playing; playBtn.innerHTML = playing ? '&#9646;&#9646; Pause' : '&#9654; Play'; }
function setSpeed(s) {
  speed = s;
  document.querySelectorAll('#hud button').forEach(b => {
    if (b.textContent.includes('x')) b.className = '';
  });
  event.target.className = 'active';
}

// --- Scene ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 100);
camera.position.set(0.65, 1.0, 3.0);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(0.65, 0.7, 0);
controls.update();
scene.add(new THREE.GridHelper(6, 30, 0x333355, 0x222244));
scene.add(new THREE.AmbientLight(0x606060));
const dl = new THREE.DirectionalLight(0xffffff, 0.7);
dl.position.set(3, 5, 4);
scene.add(dl);

// --- Build skeleton groups ---
function buildSkel(skel, showLabels) {
  const g = { group: new THREE.Group(), spheres: [], lines: [], lineGeos: [] };
  const J = skel.names.length;
  for (let j = 0; j < J; j++) {
    const mat = new THREE.MeshPhongMaterial({color: skel.colors[j]});
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.018, 6, 6), mat);
    const p = skel.frames[0][j];
    mesh.position.set(p[0], p[1], p[2]);
    mesh.userData = {name: skel.names[j], skel: skel.title};
    g.group.add(mesh);
    g.spheres.push(mesh);
    if (skel.parents[j] >= 0) {
      const geo = new THREE.BufferGeometry();
      const pos = new Float32Array(6);
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({color: skel.colors[j]}));
      g.group.add(line);
      g.lines.push({line, geo, child: j, parent: skel.parents[j]});
    }
    if (showLabels) {
      const c = document.createElement('canvas'); c.width=256; c.height=36;
      const ctx = c.getContext('2d'); ctx.font='18px monospace'; ctx.fillStyle=skel.colors[j];
      ctx.fillText(skel.names[j], 2, 24);
      const sp = new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true}));
      sp.position.set(p[0]+0.03, p[1]+0.03, p[2]);
      sp.scale.set(0.15, 0.022, 1);
      sp.userData = {jointIdx: j};
      g.group.add(sp);
      if (!g.labels) g.labels = [];
      g.labels.push(sp);
    }
  }
  // Title
  const tc = document.createElement('canvas'); tc.width=512; tc.height=48;
  const tctx = tc.getContext('2d'); tctx.font='bold 26px monospace'; tctx.fillStyle='#fff';
  tctx.textAlign='center'; tctx.fillText(skel.title, 256, 34);
  const tsp = new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(tc),transparent:true}));
  tsp.position.set(skel.frames[0][0][0], 1.7, 0);
  tsp.scale.set(0.5, 0.05, 1);
  g.group.add(tsp); g.titleSprite = tsp;
  scene.add(g.group);
  return g;
}

const srcG = buildSkel(D.source, true);
const tgtG = buildSkel(D.target, false);
const allMeshes = [...srcG.spheres, ...tgtG.spheres];

function updateSkel(g, skel, frame) {
  const f = skel.frames[frame];
  for (let j = 0; j < g.spheres.length; j++) {
    g.spheres[j].position.set(f[j][0], f[j][1], f[j][2]);
  }
  for (const {geo, child, parent} of g.lines) {
    const pa = geo.attributes.position.array;
    pa[0]=f[parent][0]; pa[1]=f[parent][1]; pa[2]=f[parent][2];
    pa[3]=f[child][0];  pa[4]=f[child][1];  pa[5]=f[child][2];
    geo.attributes.position.needsUpdate = true;
  }
  if (g.labels) {
    for (const sp of g.labels) {
      const j = sp.userData.jointIdx;
      sp.position.set(f[j][0]+0.03, f[j][1]+0.03, f[j][2]);
    }
  }
}

// Initial update
updateSkel(srcG, D.source, 0);
updateSkel(tgtG, D.target, 0);

// --- Hover ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2(-9,-9);
const hoverDiv = document.getElementById('hover-info');
renderer.domElement.addEventListener('mousemove', e => {
  mouse.x = (e.clientX/innerWidth)*2-1; mouse.y = -(e.clientY/innerHeight)*2+1;
});

// --- Keyboard ---
document.addEventListener('keydown', e => {
  if (e.key === ' ') { togglePlay(); e.preventDefault(); }
  if (e.key === 'ArrowRight') { setFrame(Math.min(currentFrame+1, D.numFrames-1)); }
  if (e.key === 'ArrowLeft') { setFrame(Math.max(currentFrame-1, 0)); }
});
slider.addEventListener('input', () => setFrame(parseInt(slider.value)));

function setFrame(f) {
  currentFrame = f; accumTime = f / D.fps;
  slider.value = f; frameLabel.textContent = `${f} / ${D.numFrames-1}`;
  updateSkel(srcG, D.source, f);
  updateSkel(tgtG, D.target, f);
}

// --- Animate ---
let lastTime = performance.now();
function animate(now) {
  requestAnimationFrame(animate);
  const dt = (now - lastTime) / 1000;
  lastTime = now;

  if (playing) {
    accumTime += dt * speed;
    let f = Math.floor(accumTime * D.fps) % D.numFrames;
    if (f < 0) f += D.numFrames;
    if (f !== currentFrame) setFrame(f);
  }

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(allMeshes);
  hoverDiv.textContent = hits.length > 0
    ? `${hits[0].object.userData.skel}: ${hits[0].object.userData.name}`
    : 'Hover joints for names';

  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

window.addEventListener('resize', () => {
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
</script>
</body>
</html>"""


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_motion.py <motion.npy> [--config configs/isaac_to_banana.json]")
        sys.exit(1)

    motion_path = sys.argv[1]
    config_path = "configs/isaac_to_banana.json"
    for i, a in enumerate(sys.argv):
        if a == "--config" and i + 1 < len(sys.argv):
            config_path = sys.argv[i + 1]

    # --- Load ---
    print(f"Loading motion: {motion_path}")
    motion = load_npy_motion(motion_path)
    config = load_json(config_path)
    target = load_json(config["target_skeleton"])

    src_names = motion["bone_names"]
    tgt_names = target["bone_names"]
    src_parents = motion["parent_indices"]
    tgt_parents = target["parent_indices"]

    T = motion["num_frames"]
    J_src = motion["num_joints"]

    # Source A-pose bind (in Isaac coords)
    src_bind_pos, src_bind_rot = make_isaac_a_pose_bind(
        src_names, motion["local_translation"]
    )
    # Convert to Unity coords
    src_bind_pos_u = [isaac_pos_to_unity(p) for p in src_bind_pos]
    src_bind_rot_u = [isaac_quat_to_unity(q) for q in src_bind_rot]

    # Target bind
    tgt_bind_pos = [v3(d) for d in target["target_bind_local_pos"]]
    tgt_bind_rot = [q4(d) for d in target["target_bind_local_rot"]]

    # Bone mapping
    mapping = build_bone_mapping(config, src_names, tgt_names)

    # Root scale (height ratio)
    src_wp0, _ = fk_full(src_parents, src_bind_pos_u, src_bind_rot_u)
    tgt_wp0, _ = fk_full(tgt_parents, tgt_bind_pos, tgt_bind_rot)
    src_h = max(p[1] for p in src_wp0) - min(p[1] for p in src_wp0)
    tgt_h = max(p[1] for p in tgt_wp0) - min(p[1] for p in tgt_wp0)
    root_scale = tgt_h / src_h if src_h > 1e-6 else 1.0

    print(f"Source: {J_src} bones, {T} frames @ {motion['fps']} FPS "
          f"({T/motion['fps']:.1f}s)")
    print(f"Target: {len(tgt_names)} bones")
    print(f"Mappings: {len(mapping)}")
    print(f"Root scale: {root_scale:.6f}")

    # --- Retarget every frame ---
    frames_json = []
    src_world_all = []
    tgt_world_all = []

    print(f"Retargeting {T} frames...")
    for t in range(T):
        if t % 100 == 0 and t > 0:
            print(f"  {t}/{T}")

        # Source local rotations for this frame, converted to Unity
        src_cur = [quat_normalize(motion["rotation"][t, j]) for j in range(J_src)]
        src_cur_u = [isaac_quat_to_unity(q) for q in src_cur]

        # Retarget rotations
        tgt_lr = retarget_frame(
            src_parents, src_bind_rot_u, src_cur_u,
            tgt_parents, tgt_bind_rot, mapping,
        )

        # Root position
        root_isaac = motion["root_translation"][t]
        root_unity = isaac_pos_to_unity(root_isaac) * root_scale

        frames_json.append({
            "root_pos": v3d(root_unity),
            "target_local_rot": [q4d(q) for q in tgt_lr],
        })

        # --- World positions for visualization ---
        # Source: use animation root_pos + bind local offsets for non-root
        src_lp = [p.copy() for p in src_bind_pos]
        src_lp[0] = root_isaac
        src_lp_u = [isaac_pos_to_unity(p) for p in src_lp]
        src_wp, _ = fk_full(src_parents, src_lp_u, src_cur_u)
        src_world_all.append(src_wp)

        # Target: retargeted rotations + bind positions with root override
        tgt_lp = [p.copy() for p in tgt_bind_pos]
        tgt_lp[0] = root_unity
        tgt_wp, _ = fk_full(tgt_parents, tgt_lp, tgt_lr)
        tgt_world_all.append(tgt_wp)

    print(f"Done retargeting.")

    # --- Save Unity JSON ---
    base = os.path.splitext(os.path.basename(motion_path))[0]
    out_json = f"exports/motion_data/{base}_unity.json"
    save_json(out_json, {
        "target_name": target["target_name"],
        "bone_names": tgt_names,
        "parent_indices": tgt_parents,
        "target_bind_local_pos": target["target_bind_local_pos"],
        "target_bind_local_rot": target["target_bind_local_rot"],
        "fps": motion["fps"],
        "frames": frames_json,
    })
    print(f"Saved Unity clip:  {out_json}")

    # --- Generate animation HTML ---
    out_html = f"exports/{base}_viewer.html"
    generate_animation_html(
        out_html,
        src_names, src_parents, src_world_all,
        tgt_names, tgt_parents, tgt_world_all,
        motion["fps"],
    )
    print(f"Saved viewer:      {out_html}")
    print(f"\nOpen {out_html} in a browser to preview.")

    try:
        webbrowser.open(f"file://{os.path.abspath(out_html)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
