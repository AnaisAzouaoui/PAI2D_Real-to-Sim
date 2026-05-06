import os
import sys
import json
import subprocess
import traceback
from PyQt6.QtCore import QObject, pyqtSignal
from pipeline.versions.v1_llm_prim.object_recognition.object_rec_v1_1_embedding import object_rec as _object_rec_v11
from pipeline.versions.v1_llm_prim.object_recognition.object_rec_v1_llm import object_rec as _object_rec_v1
from pipeline.versions.v1_llm_prim.pipeline_v1_1_1 import (object_rec as _object_rec_v111,place_scene as _place_scene_v111,modify_scene as modify_scene_v111,)
from pipeline.utils.catalogue import objets_list, OBJETS_DIR
from pipeline.utils.ollama_client import suggest_alternatives, classify_intent
from pipeline.versions.v2_llm_only.pipeline_v2_llm import object_dim_quat, modify_scene as modify_scene_v2
from pipeline.versions.v1_llm_prim.placement.placement_v1_relations import object_relations, modify_scene as modify_scene_v1
from pipeline.itemSpec import loadScale
from pipeline.versions.v1_llm_prim.placement.placement_v1_distances_orientations import orientation
import tempfile

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def build_scene_subprocess(items, relations, orientations):
    """Lance buildScene dans un subprocess pour eviter le crash NSWindow sur macOS."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(items, f); items_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(relations, f); rel_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(orientations, f); ori_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        out_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, 'run_build_scene.py'),
             items_path, rel_path, ori_path, out_path],
            cwd=SRC_DIR
        )
        if proc.returncode != 0:
            raise RuntimeError("run_build_scene.py a echoue")
        with open(out_path) as f:
            return json.load(f)
    finally:
        for p in [items_path, rel_path, ori_path, out_path]:
            try: os.unlink(p)
            except OSError: pass


def get_genesis_dims_subprocess(items, orientations):
    """Lance get_genesis_dimensions dans un subprocess pour eviter le crash NSWindow sur macOS."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(items, f); items_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(orientations, f); ori_path = f.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        out_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(SRC_DIR, 'run_get_genesis_dims.py'),
             items_path, ori_path, out_path],
            cwd=SRC_DIR
        )
        if proc.returncode != 0:
            raise RuntimeError("run_get_genesis_dims.py a echoue")
        with open(out_path) as f:
            return json.load(f)
    finally:
        for p in [items_path, ori_path, out_path]:
            try: os.unlink(p)
            except OSError: pass


# "V1" : object_rec via LLM, placement via sceneBuilding
# "V1.1" : object_rec via embeddings, placement via sceneBuilding
# "V1.1.1" : object_rec + relations + orientations via phi3-scene (finetuned), scales via phi3:mini
# "V2": placement et modification 100% LLM direct (sans sceneBuilding)
PIPELINE_VERSION = "V1.1.1"

# on choisit object_rec selon la version
if PIPELINE_VERSION == "V1.1.1":
    object_rec = _object_rec_v111
elif PIPELINE_VERSION == "V1.1":
    object_rec = _object_rec_v11
else:
    object_rec = _object_rec_v1

def set_text_version(version):
    global PIPELINE_VERSION, object_rec
    PIPELINE_VERSION = version
    if version == "V1.1.1":
        object_rec = _object_rec_v111
    elif version == "V1.1":
        object_rec = _object_rec_v11
    else:
        object_rec = _object_rec_v1
    print(f"[VERSION] Pipeline texte -> {version}")


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_OUTPUT_FILE = os.path.join(SRC_DIR, '..', 'scenes', 'scene_generee.json')


def _place_objects(prompt, objet_reconnus, relations_corrigees=None):
    """Place les objets selon la version du pipeline choisie."""
    if PIPELINE_VERSION == "V1.1.1":
        if relations_corrigees is not None:
            from pipeline.versions.v1_llm_prim.pipeline_v1_1_1 import phi3_cache
            cached = dict(phi3_cache.get(prompt, {}))
            cached["relations"] = relations_corrigees.get("relations", [])
            cached["root"]      = relations_corrigees.get("root", "")
            phi3_cache[prompt] = cached
        return _place_scene_v111(prompt, objet_reconnus, build_scene_fn=build_scene_subprocess)

    if PIPELINE_VERSION in ("V1", "V1.1"):
        if relations_corrigees is not None:
            relations_data = relations_corrigees
        else:
            relations_data = object_relations(prompt, objet_reconnus)
            
        root_id = relations_data.get("root", "")
        relations = relations_data.get("relations", [])

        items = []
        for label, info in objet_reconnus.items():
            items.append({
                "id": label,
                "urdf": info["urdf"],
                "path": info.get("path", ""),
                "dimensions": info.get("dimensions"),
                "scale": loadScale({"urdf": info["urdf"]})["scale"],
                "root": (label == root_id),
            })

        # Si le root retourné ne correspond à aucun item, marquer le premier
        if root_id not in objet_reconnus and items:
            print(f"[V1] root '{root_id}' introuvable, fallback sur '{items[0]['id']}'")
            items[0]["root"] = True

        orientations_data = orientation(prompt, objet_reconnus)
        print("items :", items, "  |relations : ", relations," |orientations :" , orientations_data)
        items = build_scene_subprocess(items, relations, orientations_data)

        for item in items:
            pos = item.get("pos", (0, 0, 0))
            item["pos"] = list(pos)
            quat = item.get("quat", (0, 0, 0, 1))
            x, y, z, w = quat
            item["quat"] = [w, x, y, z]

        return items
    else:
        return object_dim_quat(prompt, objet_reconnus)


def _modify_scene(prompt, current_objects_json, objet_reconnus):
    """Modifie la scene selon la version active du pipeline."""
    if PIPELINE_VERSION == "V1.1.1":
        return modify_scene_v111(prompt, current_objects_json, objet_reconnus, build_scene_fn=build_scene_subprocess)
    if PIPELINE_VERSION in ("V1", "V1.1"):
        return modify_scene_v1(prompt, current_objects_json, objet_reconnus, build_scene_fn=build_scene_subprocess)
    return modify_scene_v2(prompt, current_objects_json, objet_reconnus)


def resolve_urdf_path(path):
    """Resout le chemin vers le fichier exploitable (urdf ou mesh)"""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ["mobility.urdf", "kinbody.xml", "textured.obj", "nontextured.stl", "nontextured.ply"]:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return candidate
        google_16k = os.path.join(path, "google_16k")
        if os.path.isdir(google_16k):
            for name in ["textured.obj", "nontextured.stl", "nontextured.ply"]:
                candidate = os.path.join(google_16k, name)
                if os.path.exists(candidate):
                    return candidate
    return path


def postprocess_objects(objetsList, objet_reconnus):
    """Post-traitement commun : resoudre les paths, convertir les quaternions, completer les champs manquants, sauvegarder."""
    print(f"[POSTPROCESS] Post-traitement de {len(objetsList)} objets...")
    for obj in objetsList:
        info = objet_reconnus.get(obj.get("id"), {})

        # path : ajouter depuis objet_reconnus si absent, puis resoudre vers fichier
        if not obj.get("path") and info.get("path"):
            obj["path"] = info["path"]
        # fallback: si le id du LLM ne correspond pas aux cles d'objet_reconnus, chercher par urdf
        if not obj.get("path"):
            urdf = obj.get("urdf", "")
            catalog = objets_list(OBJETS_DIR)
            if urdf in catalog:
                obj["path"] = catalog[urdf]["path"]
        if obj.get("path"):
            obj["path"] = resolve_urdf_path(obj["path"])

        # dimensions : injecter depuis objet_reconnus si absentes
        if not obj.get("dimensions") and info.get("dimensions"):
            obj["dimensions"] = info["dimensions"]

        # root : injecter False si absent (seul le root explicite vaut True)
        if "root" not in obj:
            obj["root"] = info.get("root", False)

        # quat : le LLM retourne [w,x,y,z], scipy attend [x,y,z,w]
        if "quat" in obj and len(obj["quat"]) == 4:
            w, x, y, z = obj["quat"]
            obj["quat"] = [x, y, z, w]

    with open(SCENE_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(objetsList, f, indent=2, ensure_ascii=False)
    print(f"[POSTPROCESS] JSON sauvegarde -> {SCENE_OUTPUT_FILE}")

    return objetsList


class RecognitionWorker(QObject):
    """Phase 1 : reconnaissance des objets + suggestion d'alternatives pour les non reconnus."""
    status_update = pyqtSignal(str)
    recognition_done = pyqtSignal(dict, list, dict)  # objet_reconnus, non_reconnus, alternatives_map
    scene_ready = pyqtSignal(list)  # si tout est reconnu, on enchaine directement
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt
        

    def run(self):
        try:
            print(f"[RecognitionWorker] Demarrage reconnaissance | prompt: '{self.prompt}'")
            self.status_update.emit("Reconnaissance des objets...")
            objet_reconnus, non_reconnus = object_rec(self.prompt)

            if non_reconnus:
                # proposer des alternatives pour chaque objet non reconnu
                print(f"[RecognitionWorker] Objets non reconnus : {non_reconnus} — recherche d'alternatives...")
                self.status_update.emit("Recherche d'alternatives...")
                alternatives_map = {}
                for label in non_reconnus:
                    alts = suggest_alternatives(label)
                    print(f"[RecognitionWorker] Alternatives pour '{label}': {alts}")
                    alternatives_map[label] = alts
                self.recognition_done.emit(objet_reconnus, non_reconnus, alternatives_map)

                if not objet_reconnus:
                    # rien de reconnu du tout, on laisse l'UI gerer
                    return
                return

            if not objet_reconnus:
                self.error_occurred.emit("Aucun objet reconnu dans la description.")
                return

            # tout est reconnu, on enchaine avec le placement
            print(f"[RecognitionWorker] Tous reconnus : {list(objet_reconnus.keys())} → placement direct")
            self.status_update.emit("Calcul des positions et orientations...")
            objetsList = _place_objects(self.prompt, objet_reconnus)
            objetsList = postprocess_objects(objetsList, objet_reconnus)
            self.scene_ready.emit(objetsList)

        except (ConnectionError, RuntimeError, ValueError) as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(f"Erreur inattendue : {str(e)}")
        finally:
            self.finished.emit()


class PlacementWorker(QObject):
    """Phase 2 : calcul des positions/orientations apres resolution des objets"""
    status_update = pyqtSignal(str)
    scene_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, prompt, objet_reconnus):
        super().__init__()
        self.prompt = prompt
        self.objet_reconnus = objet_reconnus

    def run(self):
        try:
            print(f"[PlacementWorker] Calcul positions pour : {list(self.objet_reconnus.keys())}")
            self.status_update.emit("Calcul des positions et orientations...")
            objetsList = _place_objects(self.prompt, self.objet_reconnus)
            objetsList = postprocess_objects(objetsList, self.objet_reconnus)
            self.scene_ready.emit(objetsList)

        except (ConnectionError, RuntimeError, ValueError) as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(f"Erreur inattendue : {str(e)}")
        finally:
            self.finished.emit()


class ImageSceneWorker(QObject):
    """Génère une scène à partir d'images via le pipeline V3."""
    status_update = pyqtSignal(str)
    scene_ready   = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    finished      = pyqtSignal()

    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths

    def run(self):
        try:
            print(f"[ImageSceneWorker] Demarrage | {len(self.image_paths)} image(s): {self.image_paths}")
            self.status_update.emit("Analyse de l'image en cours (V3)...")
            print("[ImageSceneWorker] Import du pipeline V3...")
            from pipeline.versions.v3_image.pipeline_v3_image import scene_from_image
            print("[ImageSceneWorker] Import OK, appel de scene_from_image...")
            objetsList = scene_from_image(self.image_paths, dims_fn=get_genesis_dims_subprocess)
            if not objetsList:
                self.error_occurred.emit("Aucun objet reconnu dans l'image.")
                return
            for obj in objetsList:
                if "path" in obj:
                    obj["path"] = resolve_urdf_path(obj["path"])
            with open(SCENE_OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(objetsList, f, indent=2, ensure_ascii=False)
            print(f"[ImageSceneWorker] JSON sauvegarde {SCENE_OUTPUT_FILE}")
            self.scene_ready.emit(objetsList)
        except (ConnectionError, RuntimeError, ValueError) as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(f"Erreur inattendue : {str(e)}")
        finally:
            self.finished.emit()


class ModifySceneWorker(QObject):
    """Modifie une scene existante selon le prompt utilisateur."""
    status_update = pyqtSignal(str)
    scene_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, prompt, current_objects_json, objet_reconnus):
        super().__init__()
        self.prompt = prompt
        self.current_objects_json = current_objects_json
        self.objet_reconnus = objet_reconnus

    def run(self):
        try:
            print(f"[ModifySceneWorker] Modification de la scene | prompt: '{self.prompt}'")
            print(f"[ModifySceneWorker] Scene actuelle : {self.current_objects_json[:200]}...")
            self.status_update.emit("Modification de la scene...")
            objetsList = _modify_scene(self.prompt, self.current_objects_json, self.objet_reconnus)
            objetsList = postprocess_objects(objetsList, self.objet_reconnus)
            self.scene_ready.emit(objetsList)

        except (ConnectionError, RuntimeError, ValueError) as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(f"Erreur inattendue : {str(e)}")
        finally:
            self.finished.emit()


class ValidationWorker(QObject):
    """Lance la boucle de validation (physique + VLM) sur la scene courante."""
    status_update = pyqtSignal(str)
    scene_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, json_file, prompt=None, image_path=None):
        super().__init__()
        self.json_file = json_file
        self.prompt = prompt
        self.image_path = image_path

    def run(self):
        try:
            script = os.path.join(SRC_DIR, 'run_validation.py')
            if self.image_path:
                print(f"[ValidationWorker] Mode image | ref: '{self.image_path}'")
                self.status_update.emit("Validation avec image de reference...")
                args = [sys.executable, script, 'image', self.json_file, self.image_path]
            else:
                print(f"[ValidationWorker] Mode prompt | prompt: '{self.prompt}'")
                self.status_update.emit("Validation semantique...")
                args = [sys.executable, script, 'prompt', self.json_file, self.prompt]

            proc = subprocess.Popen(args, cwd=SRC_DIR)
            proc.wait()

            if proc.returncode == 0:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                self.scene_ready.emit(result)
            else:
                self.error_occurred.emit("La validation a echoue (voir logs du sous-processus).")
        except (ConnectionError, RuntimeError, ValueError) as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(f"Erreur validation : {str(e)}")
        finally:
            self.finished.emit()


class IntentWorker(QObject):
    """Classifie le prompt en 'new_scene' ou 'modify_scene'."""
    intent_classified = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, prompt, scene_summary):
        super().__init__()
        self.prompt = prompt
        self.scene_summary = scene_summary

    def run(self):
        try:
            print(f"[IntentWorker] Classification de l'intent | prompt: '{self.prompt}'")
            intent = classify_intent(self.prompt, self.scene_summary)
            print(f"[IntentWorker] Intent retourne : '{intent}'")
            self.intent_classified.emit(intent)
        except (ConnectionError, RuntimeError, ValueError) as e:
            self.error_occurred.emit(str(e))
        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(f"Erreur inattendue : {str(e)}")
        finally:
            self.finished.emit()
