import os
import json
import functools

OBJETS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "objets"))


#---------------------------------------------------------------------
# fonctions auxilieres
#---------------------------------------------------------------------

def bounding_box_from_obj(obj_file):
  "obtenir manuellement la bbox comme les objets ycb on en pas"
  xs, ys, zs = [], [], []
  with open(obj_file) as f:
      for line in f:
          if line.startswith("v "):
              v, x, y, z = line.split()
              xs.append(float(x))
              ys.append(float(y))
              zs.append(float(z))
  if not xs:
      return None
  return {"min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]}


def get_dimensions(obj_path):
  "dans le nom sah"
  bbox_path = os.path.join(obj_path, "bounding_box.json")
  if not os.path.isfile(bbox_path):
      # objets YCB : calcul depuis le mesh et sauvegarde pour la prochaine fois
      google_16k = os.path.join(obj_path, "google_16k")
      if not os.path.isdir(google_16k):
          return None
      obj_file = os.path.join(google_16k, "textured.obj")
      if not os.path.exists(obj_file):
          candidates = [f for f in os.listdir(google_16k) if f.endswith(".obj")]
          if not candidates:
              return None
          obj_file = os.path.join(google_16k, candidates[0])
      bbox = bounding_box_from_obj(obj_file)
      if bbox is None:
          return None
      with open(bbox_path, "w") as f:
          json.dump(bbox, f)
      print(f"[get_dimensions] bbox genere pour {os.path.basename(obj_path)}")
  else:
      with open(bbox_path) as f:
          bbox = json.load(f)

  return [round(bbox["max"][i] - bbox["min"][i], 4) for i in range(3)]


@functools.lru_cache(maxsize=1)
def objets_list(dirpath=OBJETS_DIR):
  """liste des objets avec leur nom, leur dimension, et leur path — cache pour pas refaire le calcule"""
  result = {}
  with os.scandir(dirpath) as it:
    for entry in it:
      if entry.name.startswith(".") or not entry.is_dir():
        continue
      nom_obj = entry.name.replace("_", " ")
      result[entry.name] = {
        "name": nom_obj,
        "path": entry.path,
        "dimensions": get_dimensions(entry.path)
      }
  return result

#print(objets_list(OBJETS_DIR).keys())


@functools.lru_cache(maxsize=1)
def objects_desc(dirpath=OBJETS_DIR):
  """chaine de description du catalogue — cache pour pas refaire le calcule"""
  objects_data = objets_list(dirpath)
  return "\n".join(
    f'- urdf: "{k}" | nom: "{v["name"].strip()}" | dimensions: {v["dimensions"]} m'
    for k, v in objects_data.items()
    if v["dimensions"] is not None
  )

# print(objects_desc(OBJETS_DIR))
