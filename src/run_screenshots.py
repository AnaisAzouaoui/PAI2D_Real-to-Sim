import sys
import os
import json

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from simulation.simulationGenesis import make_morph
import genesis as gs
from PIL import Image

CAMERAS = [
    ("perspective", dict(pos=(3.5, 0.0, 2.5), lookat=(0, 0, 0.5), fov=30)),
    ("top",dict(pos=(0.0, 0.0, 4.0), lookat=(0, 0, 0), fov=40)),
    ("side",dict(pos=(0.0, 3.5, 1.0), lookat=(0, 0, 0.5),fov=30)),
    ("side2",dict(pos=(3.5, 0.0, 0.5), lookat=(0, 0, 0.5), fov=30)),
]

PHYSICS_STEPS = 80
def render_screenshots(scene_json_path, output_dir):
    with open(scene_json_path, encoding="utf-8") as f:
        objects_list = json.load(f)
    os.makedirs(output_dir, exist_ok=True)
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
    for name, cam_kwargs in CAMERAS:
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

    for name, _ in CAMERAS:
        rgb, _, _, _ = cameras[name].render(rgb=True)
        img = Image.fromarray(rgb)
        out_path = os.path.join(output_dir, f"{name}.png")
        img.save(out_path)
        print(f"[run_screenshots] {name}.png -> {out_path}")


if __name__ == "__main__":
    render_screenshots(sys.argv[1], sys.argv[2])
