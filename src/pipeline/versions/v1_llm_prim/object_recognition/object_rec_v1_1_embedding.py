from pipeline.utils.ollama_client import validate_json_response, call_llm
from pipeline.utils.catalogue import objets_list
import numpy as np


# ---------------------------------------------------------------------
# EMBEDDING MATCHER
# ---------------------------------------------------------------------
# Principe : on represente chaque objet du catalogue par un vecteur de 768 nombres
# => multilingual-e5-base (Microsoft) est entraine pour la recuperation cross-lingue
# "four" (fr) et "oven" (en) donnent des vecteurs tres proches grace aux prefixes query/passage
# Quand on recoit un label (du type "frigo"), on le convertit aussi en vecteur,
# puis on mesure la proximite entre ce vecteur et chaque vecteur du catalogue.
# L'objet avec le vecteur le plus proche gagne.
# Ca marche en francais ET en anglais parce que le modele est multilingue :
# "fridge" et "frigo" donnent des vecteurs tres proches meme si les mots sont differents >:DDD
# ---------------------------------------------------------------------

# en dessous de ce score de similarite, on considere que l'objet n'est pas dans le catalogue
SEUIL = 0.55

# ces trois variables sont initialisees a None et remplies au premier appel
# cn  charge le modele qu'une seule fois et on le garde en memoire pour les appels suivants
# histoire de ralentir nos 15 minutes de temps d'exec >:D
embed_model = None  # modele SentenceTransformer
catalogue_vecs = None  # matrice (N_objets, 768), un vecteur par objet du catalogue
catalogue_keys = None  # liste des cles URDF, dans le meme ordre que les lignes de _catalogue_vecs pour le lien

def model():
    """retourne le modele d'embeddings, en le chargeant si c'est le premier appel"""
    global embed_model
    if embed_model is None:
        print("[embedding] import torch + sentence_transformers (peut prendre 30s)...")
        from sentence_transformers import SentenceTransformer
        print("[embedding] chargement du modele...")
        # force CPU : le GPU Metal est sature par Ollama, et MiniLM 384D
        # tourne en quelques ms sur CPU (25 objets + 1 query par appel)
        embed_model = SentenceTransformer('intfloat/multilingual-e5-base', device='cpu')
    return embed_model


def catalogue_embeddings():
    """encode tous les noms du catalogue en vecteurs et les met en cachet"""
    global catalogue_vecs, catalogue_keys
    if catalogue_vecs is None:
        data = objets_list()  # dict {urdf_key: {name, path, dimensions, ...}}

        # on garde les cles URDF dans une liste ordonnee pour retrouver l'objet
        # a partir d'un indice apres argmax
        catalogue_keys = list(data.keys())

        # on encode les noms lisibles ("refrigerateur", "banane"...)
        names = [data[k]["name"].strip() for k in catalogue_keys]

        # e5 : prefixe "passage: " pour les documents du catalogue
        catalogue_vecs = model().encode(["passage: " + n for n in names])

        print(f"[embedding] catalogue encode : {len(catalogue_keys)} objets")
    return catalogue_vecs, catalogue_keys


def cosinus_score(query_vec, catalogue_vecs):
    """ calcule la similarite cosinus entre query_vec et chaque ligne de catalogue_vecs
    retourne un vecteur de N scores, un par objet du catalogue

    on normalise d'abord les vecteurs (division par leur norme)
    apres normalisation, le produit scalaire entre deux vecteurs = leur cosinus
    cosinus = 1.0 si les vecteurs pointent dans la meme direction (meme sens)
    cosinus = 0.0 si les vecteurs sont perpendiculaires (pas de lien)
    cosinus = -1.0 si les vecteurs pointent en sens oppose"""

    q = query_vec / np.linalg.norm(query_vec)  # (768,)
    # keepdims=True garde la dimension pour que la division broadcast correctement
    c = catalogue_vecs / np.linalg.norm(catalogue_vecs, axis=1, keepdims=True)  # shape (N, 768)

    # (N, 768) @ (768,) = (N,)
    return c @ q # donne un score de similarite par objet du catalogue


# ---------------------------------------------------------------------
# EXTRACTION DES LABELS
# ---------------------------------------------------------------------
# le LLM fait juste du parsing de texte : il lit la phrase et
# liste les noms d'objets qu'il y trouve, en gerant les pluriels and that's it
# His pea brain can manage >:D
# ---------------------------------------------------------------------

def extract_labels(prompt):
    system_prompt = """You are a JSON API. Extract every distinct object from the user's scene description.

    Handle plurals by repeating the label: "three chairs" -> ["chaise", "chaise", "chaise"]
    Keep labels in the original language (French or English), lowercase.

    OUTPUT FORMAT — return ONLY: {"objects": ["label1", "label2", ...]}

    EXAMPLES:
    Input: "je veux un mug sur une table avec trois chaises"
    Output: {"objects": ["mug", "table", "chaise", "chaise", "chaise"]}

    Input: "un frigo et deux poubelles"
    Output: {"objects": ["frigo", "poubelle", "poubelle"]}

    Input: "a fridge next to a washing machine"
    Output: {"objects": ["fridge", "washing machine"]}

    Raw JSON only, no markdown, no comments."""

    result = call_llm(system_prompt, prompt)
    return result.get("objects", [])


# ---------------------------------------------------------------------
# RECONNAISSANCE D'OBJETS
# ---------------------------------------------------------------------
# Remplace object_rec de object_rec_v1_llm.
# Pipeline :
#   1. LLM extrait les labels bruts de la phrase ("frigo", "table", "chaise"...)
#   2. Pour chaque label, on calcule le cosinus contre tous les objets du catalogue
#   3. Si le meilleur score depasse seuil objet reconnu, sinon non reconnu
# zero hallucination possible believe
# ---------------------------------------------------------------------

def object_rec(prompt=None):
    if not prompt:
        return {}, []

    if isinstance(prompt, list):
        labels = prompt
    else:
        labels = extract_labels(prompt)
    print(f"[object_rec] labels : {labels}")

    # on recupere les vecteurs du catalogue
    catalogue_vecs, catalogue_keys = catalogue_embeddings()

    # on recupere aussi le dict complet pour avoir path et dimensions
    objects_data = objets_list()

    obj_reconnus        = {}
    objets_non_reconnus = []

    # counts sert a gerer les doublons : si on a deux "chaise"
    # le premier s'appelle "chaise" et le deuxieme "chaise_2"
    counts = {}

    for label in labels:
        # etape 2 : on encode le label en vecteur et on cherche le plus proche dans le catalogue
        # e5 : prefixe "query: " pour les requetes
        query_vec  = model().encode("query: " + label)
        scores = cosinus_score(query_vec, catalogue_vecs)
        indice_correspondant = int(np.argmax(scores))
        best_score = float(scores[indice_correspondant])

        # seuil de confiance — si le meilleur match est trop faible (pour le moment je lai mi a 0.55)
        if best_score < SEUIL:
            print(f"[object_rec] '{label}' -> score {best_score:.2f} sous le seuil -> non reconnu")
            objets_non_reconnus.append(label)
            continue

        urdf = catalogue_keys[indice_correspondant]
        base_label = label.lower().strip()
        counts[base_label] = counts.get(base_label, 0) + 1
        id_label = base_label if counts[base_label] == 1 else f"{base_label}_{counts[base_label]}"
        print(f"[object_rec] '{label}' -> '{urdf}' (score {best_score:.2f})")

        # on construit l'entree avec le meme format que l'ancien object_rec
        obj_reconnus[id_label] = {
            "urdf":       urdf,
            "path":       objects_data[urdf]["path"],
            "dimensions": objects_data[urdf]["dimensions"]
        }

    print("Reconnus:", list(obj_reconnus.keys()))
    print("Non reconnus:", objets_non_reconnus)
    return obj_reconnus, objets_non_reconnus
