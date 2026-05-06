import sys
import os
import json

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from simulation.simulationGenesis import make_morph
import genesis as gs
from PIL import Image

PHYSICS_STEPS = 80


def compute_camera_params(objects_list):
    """Calcule distance et hauteur camera selon l'etendue de la scene."""
    if not objects_list:
        return 0.0, 0.0, 3.5, 1.5, 0.8

    xs, ys, zs = [], [], []
    for obj in objects_list:
        pos = obj.get("pos", [0, 0, 0])
        dims = obj.get("dimensions", [0.3, 0.3, 0.3])
        scale = obj.get("scale", 1.0)
        r = max(dims) * scale
        xs += [pos[0] - r, pos[0] + r]
        ys += [pos[1] - r, pos[1] + r]
        zs += [pos[2] - r, pos[2] + r]

    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    max_z = max(zs)
    spread = max(max(xs) - min(xs), max(ys) - min(ys), max_z)
    dist = max(spread * 2.2, 2.5)
    cam_height = max(max_z * 1.3, 1.2)
    lookat_z = max_z / 2
    return cx, cy, dist, cam_height, lookat_z


def make_cameras(cx, cy, dist, cam_height, lookat_z):
    return [
        ("front", dict(pos=(cx + dist, cy,        cam_height), lookat=(cx, cy, lookat_z), fov=35)),
        ("back",  dict(pos=(cx - dist, cy,        cam_height), lookat=(cx, cy, lookat_z), fov=35)),
        ("left",  dict(pos=(cx,        cy + dist, cam_height), lookat=(cx, cy, lookat_z), fov=35)),
        ("right", dict(pos=(cx,        cy - dist, cam_height), lookat=(cx, cy, lookat_z), fov=35)),
    ]


def render_screenshots(scene_json_path, output_dir):
    with open(scene_json_path, encoding="utf-8") as f:
        objects_list = json.load(f)
    os.makedirs(output_dir, exist_ok=True)

    cx, cy, dist, cam_height, lookat_z = compute_camera_params(objects_list)
    cameras_def = make_cameras(cx, cy, dist, cam_height, lookat_z)

    gs.init(backend=gs.cpu)
    scene = gs.Scene(
        show_viewer=False,
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(
            dt=0.001,
            integrator=gs.integrator.Euler,
        ),
    )
    scene.add_entity(gs.morphs.Plane())

    cameras = {}
    for name, cam_kwargs in cameras_def:
        cameras[name] = scene.add_camera(res=(640, 480), **cam_kwargs)

    for obj in objects_list:
        path = obj["path"]
        pos = tuple(obj["pos"])
        quat = tuple(obj.get("quat", [0.0, 0.0, 0.0, 1.0]))
        scale = obj.get("scale", 1.0)
        try:
            scene.add_entity(
                make_morph(path, pos=pos, quat=quat, scale=scale, fixed=False),
                material=gs.materials.Rigid(rho=1000, friction=0.5),
            )
        except Exception as e:
            print(f"[run_screenshots] objet ignore ({path}): {e}")

    scene.build()

    for _ in range(PHYSICS_STEPS):
        scene.step()

    for name, _ in cameras_def:
        rgb, _, _, _ = cameras[name].render(rgb=True)
        img = Image.fromarray(rgb)
        out_path = os.path.join(output_dir, f"{name}.png")
        img.save(out_path)
        print(f"[run_screenshots] {name}.png -> {out_path}")


if __name__ == "__main__":
    render_screenshots(sys.argv[1], sys.argv[2])
