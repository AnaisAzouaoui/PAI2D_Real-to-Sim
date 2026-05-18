"""
Lance tous les tests dans l'ordre et affiche un tableau comparatif final.

python tests/run_all_tests.py                                                      # tous, 5 runs, gpt-4o
python tests/run_all_tests.py --runs 1                                             # rapide, 1 run
python tests/run_all_tests.py --tests recognition placement                        # sans vision
python tests/run_all_tests.py --tests recognition placement --model gpt-4o   --tag gpt-4o
python tests/run_all_tests.py --tests recognition placement --model llama3.1  --tag llama3.1 --skip-v2
python tests/run_all_tests.py --vision-model llama3.2-vision                       # VLM Ollama pour V3
"""
import sys
import os
import argparse
import importlib
import time

ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_DIR = os.path.join(ROOT, "tests")
SRC = os.path.join(ROOT, "src")
for p in [SRC, ROOT, TESTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

TESTS = [
    ("recognition", "test_recognition"),
    ("placement",   "test_placement"),
    ("vision",      "test_vision"),
]

LLM_CHOICES    = ["gpt-4o", "gpt-4o-mini", "llama3.1", "llama3.2"]
VISION_CHOICES = ["gpt-4o", "gpt-4o-mini", "llama3.2-vision"]

VERSIONS_SANS_V2 = {
    "recognition": ["V1.1", "V1.1.1", "V1"],
    "placement":   ["V1.1", "V1.1.1", "V1"],
}


def set_models(model, vision_model, selected=None):
    import pipeline.utils.ollama_client as ollama_mod
    ollama_mod.LLM_MODEL = model
    print(f"[config] LLM       = {model}")
    if selected is None or "vision" in selected:
        from pipeline.versions.v3_1_vlm_refined import pipeline_v3_1_image as v3_mod
        v3_mod.VISION_MODEL = vision_model
        print(f"[config] VisionLLM = {vision_model}")


def run_all(selected=None, n_runs=5, model="gpt-4o", vision_model="gpt-4o", tag=None, skip_v2=False):
    set_models(model, vision_model, selected)
    to_run = [(name, mod) for name, mod in TESTS
              if selected is None or name in selected]

    resume = {}
    for name, module_name in to_run:
        print(f"\n{'#'*60}")
        print(f"#  {name.upper()}")
        print(f"{'#'*60}")
        t0 = time.time()
        try:
            mod = importlib.import_module(module_name)
            kwargs = {"n_runs": n_runs, "save_json": True}
            if name in ("recognition", "placement"):
                if tag:
                    kwargs["tag"] = tag
                if skip_v2 and name in VERSIONS_SANS_V2:
                    kwargs["only_versions"] = VERSIONS_SANS_V2[name]
            result = mod.run_all(**kwargs)
            resume[name] = {"status": "OK", "wall_time": round(time.time()-t0, 1), "result": result}
        except Exception as e:
            resume[name] = {"status": f"ERREUR: {e}", "wall_time": round(time.time()-t0, 1), "result": None}
            print(f"\n[{name}] ERREUR : {e}")

    print(f"\n{'='*50}")
    print(f"  RESUME")
    print(f"{'='*50}")
    print(f"{'Test':<14} {'Statut':<20} {'Temps':>8}")
    print(f"{'-'*55}")
    for name, s in resume.items():
        print(f"{name:<14} {s['status'][:19]:<20} {s['wall_time']:>7.1f}s")
    print(f"{'='*50}\n")

    return resume


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", nargs="+", choices=[n for n, _ in TESTS], help="Tests a lancer (defaut: tous)")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", choices=LLM_CHOICES, default="gpt-4o",
                        help="LLM pour V1.1/V1.1.1/V1 (defaut: gpt-4o)")
    parser.add_argument("--vision-model", dest="vision_model",
                        choices=VISION_CHOICES, default="gpt-4o",
                        help="VLM pour V3 (defaut: gpt-4o)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Suffixe pour nommer les versions ex: gpt-4o ou llama3.1")
    parser.add_argument("--skip-v2", action="store_true",
                        help="Ne pas relancer V2 (phi3-scene), reutilise les resultats existants")
    args = parser.parse_args()
    run_all(selected=args.tests, n_runs=args.runs,
            model=args.model, vision_model=args.vision_model, tag=args.tag, skip_v2=args.skip_v2)
