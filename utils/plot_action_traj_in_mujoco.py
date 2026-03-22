"""
Utilities for 3D joint/EEF trajectory visualization in MuJoCo.

Provides:
  - FK-based trajectory computation from recorded joint positions
  - Gradient-colored trajectory overlays on MjvScene (viewer or offscreen)
  - Offscreen rendering to PNG from configurable camera views
"""

import numpy as np
import mujoco
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  Joint name constants (from model XML + meta_rby1_hdf5.json)
# ═══════════════════════════════════════════════════════════════════════════════

TORSO_JOINTS = [f"torso_{i}" for i in range(6)]
RIGHT_ARM_JOINTS = [f"right_arm_{i}" for i in range(7)]
LEFT_ARM_JOINTS = [f"left_arm_{i}" for i in range(7)]
HEAD_JOINTS = ["head_0", "head_1"]

LEFT_HAND_JOINTS = [
    "left_hand_thumb_bend_joint", "left_hand_thumb_rota_joint1", "left_hand_thumb_rota_joint2",
    "left_hand_index_bend_joint", "left_hand_index_joint1", "left_hand_index_joint2",
    "left_hand_mid_joint1", "left_hand_mid_joint2",
    "left_hand_ring_joint1", "left_hand_ring_joint2",
    "left_hand_pinky_joint1", "left_hand_pinky_joint2",
]
RIGHT_HAND_JOINTS = [
    "right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
    "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
    "right_hand_mid_joint1", "right_hand_mid_joint2",
    "right_hand_ring_joint1", "right_hand_ring_joint2",
    "right_hand_pinky_joint1", "right_hand_pinky_joint2",
]

# HDF5 robot0_joint_pos index slices
OBS_BASE = slice(0, 4)
OBS_TORSO = slice(4, 10)
OBS_RIGHT_ARM = slice(10, 17)
OBS_LEFT_ARM = slice(17, 24)
OBS_HEAD = slice(24, 26)

# Safe “ready” pose (torso + arms + head) matching teleop `reset_safe_pose` / impedance move
# to streaming-ready posture. Order: base(4) + torso(6) + right(7) + left(7) + head(2).
# Base is neutral zeros (not commanded in that snippet); hands are set separately in MuJoCo.
SAFE_POSE_ROBOT0_JOINT_26 = np.array(
    [
        # base — not in SDK body vector; neutral for visualization
        0.0,
        0.0,
        0.0,
        0.0,
        # torso (rad): torso_0 … torso_5
        0.0,
        0.5,
        -1.0,
        0.65,
        0.0,
        0.0,
        # RIGHT_READY_RAD
        -0.261799,
        -0.261799,
        0.0,
        -1.894395,
        0.0,
        0.0,
        0.0,
        # LEFT_READY_RAD
        -0.261799,
        0.261799,
        0.0,
        -1.894395,
        0.0,
        0.0,
        0.0,
        # head (rad)
        0.0,
        0.6,
    ],
    dtype=np.float64,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Low-level qpos helpers
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_qpos_addrs(model, joint_names):
    """Return list of qpos addresses for the given joint names (-1 if missing)."""
    addrs = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        addrs.append(model.jnt_qposadr[jid] if jid >= 0 else None)
    return addrs


def set_qpos_from_addrs(data, addrs, values):
    """Write values into data.qpos at resolved addresses."""
    for addr, v in zip(addrs, values):
        if addr is not None:
            data.qpos[addr] = v


def set_robot_pose(model, data, joint_pos_26, hand_left=None, hand_right=None):
    """Set the robot to a specific pose (from HDF5 obs) and run FK."""
    set_qpos_from_addrs(data, resolve_qpos_addrs(model, TORSO_JOINTS), joint_pos_26[OBS_TORSO])
    set_qpos_from_addrs(data, resolve_qpos_addrs(model, RIGHT_ARM_JOINTS), joint_pos_26[OBS_RIGHT_ARM])
    set_qpos_from_addrs(data, resolve_qpos_addrs(model, LEFT_ARM_JOINTS), joint_pos_26[OBS_LEFT_ARM])
    set_qpos_from_addrs(data, resolve_qpos_addrs(model, HEAD_JOINTS), joint_pos_26[OBS_HEAD])
    if hand_left is not None:
        set_qpos_from_addrs(data, resolve_qpos_addrs(model, LEFT_HAND_JOINTS), hand_left)
    if hand_right is not None:
        set_qpos_from_addrs(data, resolve_qpos_addrs(model, RIGHT_HAND_JOINTS), hand_right)
    mujoco.mj_forward(model, data)


# ═══════════════════════════════════════════════════════════════════════════════
#  MjvScene geometry primitives
# ═══════════════════════════════════════════════════════════════════════════════

def add_sphere(scn, pos, radius, rgba):
    """Add a sphere geom to an MjvScene."""
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def add_capsule(scn, p1, p2, radius, rgba):
    """Add a capsule between two 3D points to an MjvScene."""
    if scn.ngeom >= scn.maxgeom:
        return
    p1, p2 = np.asarray(p1, dtype=np.float64), np.asarray(p2, dtype=np.float64)
    vec = p2 - p1
    length = np.linalg.norm(vec)

    if length < 1e-6:
        add_sphere(scn, p1, radius, rgba)
        return

    z = vec / length
    ref = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack((x, y, z))

    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[radius, length / 2.0, 0],
        pos=(p1 + p2) / 2.0,
        mat=R.flatten(),
        rgba=np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def add_trajectory(scn, positions, sphere_radius, line_radius, color_start, color_end,
                   subsample=1, marker_radius=None):
    """
    Add a gradient-colored trajectory trail to an MjvScene.

    Draws small spheres at sampled points; segments between points use ``line_radius``
    (typically thicker than ``sphere_radius`` for visibility).
    RGBA interpolates along the path; alpha is in the 4th channel of color_start/end.
    """
    N = positions.shape[0]
    if N < 2:
        return
    idxs = list(range(0, N, max(1, subsample)))
    if idxs[-1] != N - 1:
        idxs.append(N - 1)
    n = len(idxs)
    cs = np.asarray(color_start, dtype=np.float32)
    ce = np.asarray(color_end, dtype=np.float32)

    for k, idx in enumerate(idxs):
        t = k / max(n - 1, 1)
        rgba = cs * (1.0 - t) + ce * t

        r = sphere_radius
        if marker_radius and (k == 0 or k == n - 1):
            r = marker_radius

        add_sphere(scn, positions[idx], r, rgba)
        if k > 0:
            add_capsule(scn, positions[idxs[k - 1]], positions[idx], line_radius, rgba)


# ═══════════════════════════════════════════════════════════════════════════════
#  FK trajectory computation
# ═══════════════════════════════════════════════════════════════════════════════

def _find_eef_body(model, last_joint_name):
    """Find the first child body of the last arm joint's body (FT sensor / hand mount)."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, last_joint_name)
    if jid < 0:
        return -1
    parent_bid = model.jnt_bodyid[jid]
    for b in range(model.nbody):
        if model.body_parentid[b] == parent_bid:
            return b
    return parent_bid


def compute_fk_trajectories(model, data, joint_pos_seq,
                            hand_left_seq=None, hand_right_seq=None):
    """
    Run forward kinematics per frame and record body world positions.

    Args:
        model, data:      MuJoCo model and data.
        joint_pos_seq:    (N, 26) robot joint positions from HDF5 obs.
        hand_left_seq:    (N, 12) or None.
        hand_right_seq:   (N, 12) or None.

    Returns:
        dict  name -> (N, 3) world positions.
        Keys: 'right_arm_J0'…'right_arm_J6', 'left_arm_J0'…'left_arm_J6',
              'right_eef', 'left_eef'.
    """
    N = joint_pos_seq.shape[0]

    # Pre-resolve all qpos addresses
    addr = {
        "torso": resolve_qpos_addrs(model, TORSO_JOINTS),
        "right": resolve_qpos_addrs(model, RIGHT_ARM_JOINTS),
        "left":  resolve_qpos_addrs(model, LEFT_ARM_JOINTS),
        "head":  resolve_qpos_addrs(model, HEAD_JOINTS),
    }
    if hand_left_seq is not None:
        addr["lhand"] = resolve_qpos_addrs(model, LEFT_HAND_JOINTS)
    if hand_right_seq is not None:
        addr["rhand"] = resolve_qpos_addrs(model, RIGHT_HAND_JOINTS)

    # Resolve tracked body IDs
    tracked = {}
    for prefix, joints in [("right_arm", RIGHT_ARM_JOINTS), ("left_arm", LEFT_ARM_JOINTS)]:
        for i, jname in enumerate(joints):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                tracked[f"{prefix}_J{i}"] = model.jnt_bodyid[jid]

    for side, last_j in [("right_eef", "right_arm_6"), ("left_eef", "left_arm_6")]:
        bid = _find_eef_body(model, last_j)
        if bid >= 0:
            tracked[side] = bid

    # Allocate
    trajectories = {name: np.zeros((N, 3)) for name in tracked}

    # FK loop
    for i in range(N):
        pos = joint_pos_seq[i]
        set_qpos_from_addrs(data, addr["torso"], pos[OBS_TORSO])
        set_qpos_from_addrs(data, addr["right"], pos[OBS_RIGHT_ARM])
        set_qpos_from_addrs(data, addr["left"],  pos[OBS_LEFT_ARM])
        set_qpos_from_addrs(data, addr["head"],  pos[OBS_HEAD])
        if "lhand" in addr:
            set_qpos_from_addrs(data, addr["lhand"], hand_left_seq[i])
        if "rhand" in addr:
            set_qpos_from_addrs(data, addr["rhand"], hand_right_seq[i])
        mujoco.mj_forward(model, data)

        for name, bid in tracked.items():
            trajectories[name][i] = data.xpos[bid].copy()

    return trajectories


# ═══════════════════════════════════════════════════════════════════════════════
#  Trajectory color scheme & scene builder
# ═══════════════════════════════════════════════════════════════════════════════

# Sphere radii (small); line_radius is thicker for segment capsules
LINK_SPHERE_RADIUS = 0.0011
LINK_LINE_RADIUS = 0.00325
LINK_MARKER_RADIUS = 0.0024

EEF_SPHERE_RADIUS = 0.00175
EEF_LINE_RADIUS = 0.00475
EEF_MARKER_RADIUS = 0.0035

# Right side: green → red (moderate alpha so trajs layer over semi-transparent robot)
RIGHT_LINK_START = [0.2, 0.85, 0.25, 0.38]
RIGHT_LINK_END   = [1.0, 0.25, 0.1, 0.62]
RIGHT_EEF_START  = [0.1, 1.0, 0.15, 0.45]
RIGHT_EEF_END    = [1.0, 0.05, 0.05, 0.78]

# Left side: cyan → purple
LEFT_LINK_START  = [0.15, 0.55, 1.0, 0.38]
LEFT_LINK_END    = [0.55, 0.1, 1.0, 0.62]
LEFT_EEF_START   = [0.1, 0.75, 1.0, 0.45]
LEFT_EEF_END     = [0.75, 0.05, 1.0, 0.78]


def distinct_rgba(index: int, alpha: float = 0.38) -> np.ndarray:
    """
    Golden-ratio hue spacing for many overlaid curves (RGB + alpha).
    """
    import colorsys

    h = (index * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.92)
    return np.array([r, g, b, alpha], dtype=np.float32)


def add_eef_trajectory_flat(
    scn,
    positions: np.ndarray,
    sphere_radius: float,
    line_radius: float,
    rgba: np.ndarray,
    subsample: int = 3,
    marker_radius=None,
):
    """Draw one EEF path with a single RGBA (no time gradient). More transparent overlays: lower rgba[3]."""
    cs = np.asarray(rgba, dtype=np.float32)
    ce = cs.copy()
    add_trajectory(
        scn, positions, sphere_radius, line_radius, cs, ce,
        subsample=subsample, marker_radius=marker_radius,
    )


def add_all_trajectories(scn, trajectories, subsample=3):
    """Add all FK trajectories to an MjvScene with color-coded gradient trails."""
    for name, traj in trajectories.items():
        is_eef = "eef" in name
        is_right = "right" in name

        if is_eef:
            sr, lr = EEF_SPHERE_RADIUS, EEF_LINE_RADIUS
            mr = EEF_MARKER_RADIUS
            cs = RIGHT_EEF_START if is_right else LEFT_EEF_START
            ce = RIGHT_EEF_END if is_right else LEFT_EEF_END
        else:
            sr, lr = LINK_SPHERE_RADIUS, LINK_LINE_RADIUS
            mr = LINK_MARKER_RADIUS
            cs = RIGHT_LINK_START if is_right else LEFT_LINK_START
            ce = RIGHT_LINK_END if is_right else LEFT_LINK_END

        add_trajectory(
            scn, traj, sr, lr, cs, ce,
            subsample=subsample, marker_radius=mr,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Robot transparency (geom RGBA)
# ═══════════════════════════════════════════════════════════════════════════════

def push_robot_alpha(model, alpha_scale=0.62):
    """
    Multiply every geom's alpha by ``alpha_scale`` (0–1). Returns backup for restore.

    Use so the static robot reads slightly transparent behind trajectory overlays.
    """
    backup = np.array(model.geom_rgba, copy=True)
    model.geom_rgba[:, 3] = np.clip(model.geom_rgba[:, 3] * float(alpha_scale), 0.0, 1.0)
    return backup


def pop_robot_alpha(model, backup):
    """Restore ``model.geom_rgba`` from ``push_robot_alpha``."""
    model.geom_rgba[:] = backup


# ═══════════════════════════════════════════════════════════════════════════════
#  Offscreen rendering → PNG
# ═══════════════════════════════════════════════════════════════════════════════

# Slightly lower lookat + steeper elevation so arms sit lower in frame
_LO = [0.0, 0.0, 0.74]

DEFAULT_CAM = dict(distance=2.6, azimuth=180, elevation=-20, lookat=_LO)

CAMERA_VIEWS = {
    "front":         dict(distance=2.6, azimuth=180, elevation=-20, lookat=_LO),
    "right_side":    dict(distance=2.6, azimuth=90,  elevation=-20, lookat=_LO),
    "left_side":     dict(distance=2.6, azimuth=270, elevation=-20, lookat=_LO),
    "top":           dict(distance=3.0, azimuth=180, elevation=-89, lookat=_LO),
    "three_quarter": dict(distance=2.6, azimuth=135, elevation=-22, lookat=_LO),
}


def make_camera(cfg=None):
    """Create an MjvCamera from a config dict."""
    cfg = cfg or DEFAULT_CAM
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = cfg.get("distance", 2.5)
    cam.azimuth = cfg.get("azimuth", 180)
    cam.elevation = cfg.get("elevation", -15)
    cam.lookat[:] = cfg.get("lookat", [0, 0, 0.74])
    return cam


def render_scene(model, data, add_geoms_fn, cam_config=None,
                 width=1920, height=1080, max_geom=20000):
    """
    Render the MuJoCo scene with custom geoms to an (H, W, 3) uint8 RGB array.

    Args:
        add_geoms_fn:  callable(scn) that adds custom geoms to an MjvScene.
        cam_config:    dict with distance/azimuth/elevation/lookat.
        max_geom:      MjvScene capacity (raise when overlaying many trajectories).
    """
    # Temporarily enlarge offscreen framebuffer if the model default is too small
    orig_w, orig_h = model.vis.global_.offwidth, model.vis.global_.offheight
    model.vis.global_.offwidth = max(orig_w, width)
    model.vis.global_.offheight = max(orig_h, height)

    try:
        cam = make_camera(cam_config)
        renderer = mujoco.Renderer(model, height=height, width=width, max_geom=max_geom)
        renderer.update_scene(data, camera=cam)
        add_geoms_fn(renderer.scene)
        pixels = renderer.render()
        renderer.close()
        return pixels
    finally:
        model.vis.global_.offwidth = orig_w
        model.vis.global_.offheight = orig_h


def save_screenshot(model, data, add_geoms_fn, save_path, cam_config=None,
                    width=1920, height=1080, max_geom=20000):
    """Render scene with custom geoms and save as PNG."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pixels = render_scene(model, data, add_geoms_fn, cam_config, width, height, max_geom)
        import cv2
        cv2.imwrite(str(save_path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"  Offscreen render failed ({save_path.name}): {e}")
        return
    print(f"  Saved: {save_path}")


def save_all_views(model, data, add_geoms_fn, output_dir, prefix="traj",
                   views=None, width=1920, height=1080, max_geom=20000):
    """Save PNG screenshots from multiple camera angles."""
    views = views or CAMERA_VIEWS
    output_dir = Path(output_dir)
    for name, cfg in views.items():
        save_screenshot(model, data, add_geoms_fn,
                        output_dir / f"{prefix}_{name}.png", cfg, width, height, max_geom)
