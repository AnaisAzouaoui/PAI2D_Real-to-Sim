"""
lance python tests/generate_formulaire_images.py --exclude table
pour faire les tests sans la table, si non enleve le exclude quand y aura la table
si tu veux faire juste des pormpts specifique lance :
python tests/generate_formulaire_images.py --ids p01 (tu veux ajouter --versions v2 (ou v1) pour lancer une version specifique)
"""
import sys
import os
import json
import subprocess
import tempfile
import traceback
import argparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC  = os.path.join(ROOT, "src")
for p in [SRC, ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline.versions.v2_finetuned.pipeline_v2 import (object_rec as object_rec_v2,place_scene,phi3_cache,)
from pipeline.versions.v1_1_llm_and_primitives.object_recognition.object_rec_v1_1_1 import (object_rec as object_rec_embed,)
from pipeline.versions.v1_llm_only.pipeline_v1_llm import object_dim_quat
from pipeline_worker import build_scene_subprocess, postprocess_objects

DATA_PATH = os.path.join(ROOT, "tests", "data", "prompts_ground_truth.json")
IMAGES_BASE_DIR = os.path.join(ROOT, "tests", "images_formulaire")
RUN_SCREENSHOTS = os.path.join(SRC, "run_screenshots.py")
RUN_VALIDATION  = os.path.join(SRC, "run_validation.py")


def load_prompts():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)["prompts"]


def save_tmp_scene(objects_list):
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(objects_list, tmp, ensure_ascii=False)
    tmp.close()
    return tmp.name


def run_validation_subprocess(scene_json_path, prompt):
    result = subprocess.run([sys.executable, RUN_VALIDATION, "prompt", scene_json_path, prompt],cwd=SRC,)

def run_screenshots_subprocess(scene_json_path, output_dir):
    result = subprocess.run([sys.executable, RUN_SCREENSHOTS, scene_json_path, output_dir],cwd=SRC,)
   
def run_v2(prompt):
    phi3_cache.pop(prompt, None)
    obj_reconnus, non_reconnus = object_rec_v2(prompt)
    if not obj_reconnus:
        raise ValueError(f"Aucun objet reconnu (v2) pour: {prompt!r}")
    if non_reconnus:
        print(f"[v2] objets non reconnus (ignores): {non_reconnus}")
    items = place_scene(prompt, obj_reconnus, build_scene_fn=build_scene_subprocess)
    items = postprocess_objects(items, obj_reconnus)
    return items


def run_v1(prompt):
    obj_reconnus, non_reconnus = object_rec_embed(prompt)
    if not obj_reconnus:
        raise ValueError(f"Aucun objet reconnu (v1) pour: {prompt!r}")
    if non_reconnus:
        print(f"  [v1] objets non reconnus (ignores): {non_reconnus}")
    items = object_dim_quat(prompt, obj_reconnus)
    items = postprocess_objects(items, obj_reconnus)
    return items


def process_version(pid, prompt, version_label, run_fn):
    out_dir = os.path.join(IMAGES_BASE_DIR, pid, version_label)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{pid}/{version_label}] pipeline...")
    try:
        items = run_fn(prompt)
    except Exception as e:
        print(f"  [{pid}/{version_label}] ERREUR pipeline: {e}")
        traceback.print_exc()
        return False

    scene_path = save_tmp_scene(items)
    try:
        print(f"  [{pid}/{version_label}] boucle validation...")
        run_validation_subprocess(scene_path, prompt)
        print(f"  [{pid}/{version_label}] rendu 4 screenshots -> {out_dir}")
        run_screenshots_subprocess(scene_path, out_dir)
        print(f"  [{pid}/{version_label}] OK")
        return True
    except Exception as e:
        print(f"  [{pid}/{version_label}] ERREUR rendu: {e}")
        return False
    finally:
        try:
            os.unlink(scene_path)
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="Genere les images formulaire pour les 30 prompts")
    parser.add_argument("--ids", nargs="+", help="IDs a traiter (ex: p01 p02), tous par defaut")
    parser.add_argument("--versions", nargs="+", choices=["v2", "v1"], default=["v2", "v1"],
                        help="Versions a executer (defaut: v2 v1)")
    parser.add_argument("--exclude", nargs="+", help="Exclure les prompts contenant ces mots (ex: table chaise)")
    args = parser.parse_args()

    prompts = load_prompts()
    if args.ids:
        prompts = [p for p in prompts if p["id"] in args.ids]
    if args.exclude:
        prompts = [p for p in prompts if not any(w.lower() in p["prompt"].lower() for w in args.exclude)]

    print(f"[generate] {len(prompts)} prompt(s) x {len(args.versions)} version(s)")
    print(f"[generate] images -> {IMAGES_BASE_DIR}")

    version_fns = {"v2": run_v2, "v1": run_v1}

    ok_count  = 0
    err_count = 0

    for entry in prompts:
        pid    = entry["id"]
        prompt = entry["prompt"]
        print(f"\n{'='*60}")
        print(f"  {pid}: {prompt[:70]}")
        print(f"{'='*60}")

        for version_label in args.versions:
            success = process_version(pid, prompt, version_label, version_fns[version_label])
            if success:
                ok_count += 1
            else:
                err_count += 1

    total = ok_count + err_count
    print(f"\n[generate] Termine: {ok_count}/{total} succes, {err_count} erreurs")


if __name__ == "__main__":
    main()
