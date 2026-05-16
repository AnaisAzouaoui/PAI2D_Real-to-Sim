import base64
import io
import json
import os
import re
from pathlib import Path

import requests
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

from pipeline.utils.catalogue import objets_list, objects_desc, OBJETS_DIR
from pipeline.utils.ollama_client import URL
from pipeline.itemSpec import loadScale

load_dotenv(Path(__file__).resolve().parents[4] / ".env")

# OpenAI  -> "gpt-4o", "gpt-4o-mini"
# Ollama  -> "llama3.2-vision"
VISION_MODEL = "gpt-4o"

VISION_TIMEOUT = 600
MAX_IMAGE_SIZE = 1024

RE_JSON_ARRAY  = re.compile(r"\[.*\]", re.DOTALL)
RE_CODE_BLOCK  = re.compile(r"```(?:json)?\s*([\s\S]*?)```")

openai_client = None


def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return openai_client


def is_openai_model(model):
    return model.startswith(("gpt-", "o1-", "o3-", "o4-"))


def open_image(source):
    img = Image.open(str(source))
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def resolve_path(path):
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
        for ext in (".obj", ".stl", ".ply"):
            for f in sorted(os.listdir(path)):
                if f.endswith(ext) and "collision" not in f.lower():
                    return os.path.join(path, f)
    return path


def build_system_prompt(catalogue_desc):
    return f"""You are a strict JSON API. Your task is to analyse the image(s) of a real scene and \
produce a 3D scene description that can be used directly to reconstruct the scene in a simulator.

CATALOGUE OF AVAILABLE OBJECTS:
{catalogue_desc}

COORDINATE SYSTEM:
- x : depth axis (positive = farther from camera)
- y : lateral axis (positive = right)
- z : vertical axis (positive = up, 0 = ground)
- Origin is the centre of the scene floor. Typical scene is ~5 x 5 m.

TASK:
1. Identify every distinct object visible in the image(s) and match it to the CLOSEST entry \
in the catalogue above (using the "urdf" key).
2. Estimate the 3D position of each object centre in metres.
   - Objects resting on the floor have z = half their height (dimensions[2] / 2).
   - Objects placed on top of another have z = parent_z + parent_height/2 + own_height/2.
3. Estimate the orientation as a quaternion [x, y, z, w].
   Use [0, 0, 0, 1] when the object is upright with no rotation.
4. Set parent_id to the urdf of the object this one rests on, or null if on the floor.

OUTPUT FORMAT — return ONLY a JSON array, nothing else:
[
  {{
    "urdf": "<urdf key from catalogue>",
    "pos": [<x>, <y>, <z>],
    "quat": [<x>, <y>, <z>, <w>],
    "parent_id": "<urdf of parent>" | null
  }}
]

RULES:
- Only use urdf keys that appear in the catalogue.
- One entry per distinct instance (two chairs = two entries with different pos).
- Return valid JSON only — no comments, no extra keys."""


def call_openai_vision(images_b64, system_prompt):
    content = [{"type": "text", "text": "Analyse the image(s) and return the JSON array."}]
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "high"},
        })

    print(f"[call_vision_llm] sending {len(images_b64)} image(s) to {VISION_MODEL}...")
    response = get_openai_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        temperature=0,
        max_tokens=2048,
    )
    raw = response.choices[0].message.content
    print("[call_vision_llm] response received.")
    return parse_json_array(raw)


def call_ollama_vision(images_b64, system_prompt):
    payload = {
        "model": VISION_MODEL,
        "system": system_prompt,
        "prompt": "Analyse the image(s) and return the JSON array.",
        "images": images_b64,
        "stream": True,
        "options": {"temperature": 0, "num_predict": 2048},
    }
    print(f"[call_vision_llm] sending request to Ollama ({VISION_MODEL}, streaming)...")
    response = requests.post(URL, json=payload, stream=True, timeout=VISION_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"Ollama vision error (HTTP {response.status_code}): {response.text}")

    full_response = ""
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        full_response += chunk.get("response", "")
        if chunk.get("done"):
            break

    print("[call_vision_llm] response received.")
    return parse_json_array(full_response)


def parse_json_array(raw):
    m = RE_CODE_BLOCK.search(raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    m = RE_JSON_ARRAY.search(raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"VLM did not return a valid JSON array. Raw:\n{raw[:400]}")


def call_vision_llm(images_b64, system_prompt):
    if is_openai_model(VISION_MODEL):
        return call_openai_vision(images_b64, system_prompt)
    return call_ollama_vision(images_b64, system_prompt)


def scene_from_image(image_paths, depth_map_paths=None, dims_fn=None):
    print(f"\n[scene_from_image V3] === debut pipeline VLM-only sur {len(image_paths)} image(s) ===")

    images_b64 = [pil_to_base64(open_image(p)) for p in image_paths]

    catalogue = objets_list(OBJETS_DIR)
    catalogue_desc = objects_desc(OBJETS_DIR)
    system_prompt = build_system_prompt(catalogue_desc)

    print("[scene_from_image V3] appel VLM direct...")
    raw_items = call_vision_llm(images_b64, system_prompt)

    if not raw_items:
        print("[scene_from_image V3] aucun objet retourne par le VLM.")
        return []

    items = []
    for i, entry in enumerate(raw_items):
        urdf = entry.get("urdf", "")
        if urdf not in catalogue:
            print(f"  [V3] urdf inconnu ignore: '{urdf}'")
            continue

        cat = catalogue[urdf]
        scale_info = loadScale({"urdf": urdf})
        dimensions = cat.get("dimensions") or [1.0, 1.0, 1.0]

        pos = entry.get("pos", [0.0, 0.0, dimensions[2] / 2])
        quat = entry.get("quat", [0, 0, 0, 1])
        parent_id = entry.get("parent_id") or None

        item = {
            "id": urdf if i == 0 else f"{urdf}_{i}",
            "urdf": urdf,
            "path": resolve_path(cat["path"]),
            "dimensions": dimensions,
            "scale": scale_info.get("scale", 1.0),
            "quat": quat,
            "pos": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
            "parent_id": parent_id,
            "root": parent_id is None,
        }
        items.append(item)
        print(f"  {item['id']}: pos={item['pos']} parent={parent_id}")

    print(f"[scene_from_image V3] === pipeline termine: {len(items)} objet(s) ===\n")
    return items
