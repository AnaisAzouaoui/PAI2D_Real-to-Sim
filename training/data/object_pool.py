OBJECT_POOL = {
    # grands objets
    "lave_linge":       "grand",
    "seche_linge":      "grand",
    "frigo":            "grand",
    "congelateur":      "grand",
    "lave_vaisselle":   "grand",
    "four":             "grand",
    "micro_onde":       "grand",
    "canape":           "grand",
    "canape_angle":     "grand",
    "lit":              "grand",
    "armoire":          "grand",
    "buffet":           "grand",
    "bibliotheque":     "grand",
    "commode":          "grand",
    "piano":            "grand",
    "baignoire":        "grand",
    "table":            "grand",
    "bureau":           "grand",
    "television":       "grand",
    "voiture":          "grand",
    "moto":             "grand",
    "velo":             "grand",
    "poubelle":         "grand",
    "machine_cafe":     "grand",
    "aquarium":         "grand",
    "table_basse":      "grand",

    # objets moyens
    "chaise":           "moyen",
    "tabouret":         "moyen",
    "fauteuil":         "moyen",
    "pouf":             "moyen",
    "table_de_nuit":    "moyen",
    "lampe":            "moyen",
    "lampadaire":       "moyen",
    "radiateur":        "moyen",
    "ventilateur":      "moyen",
    "plante":           "moyen",
    "cactus":           "moyen",
    "pot_de_fleur":     "moyen",
    "valise":           "moyen",
    "sac":              "moyen",
    "carton":           "moyen",
    "panier":           "moyen",
    "corbeille":        "moyen",
    "ordinateur":       "moyen",
    "ecran":            "moyen",
    "tableau":          "moyen",
    "miroir":           "moyen",
    "horloge":          "moyen",
    "boite":            "moyen",
    "bouteille":        "moyen",
    "carafe":           "moyen",
    "vase":             "moyen",
    "coussin":          "moyen",

    # petits objets
    "mug":              "petit",
    "tasse":            "petit",
    "verre":            "petit",
    "assiette":         "petit",
    "bol":              "petit",
    "casserole":        "petit",
    "poele":            "petit",
    "banane":           "petit",
    "pomme":            "petit",
    "orange":           "petit",
    "citron":           "petit",
    "tomate":           "petit",
    "pain":             "petit",
    "gateau":           "petit",
    "livre":            "petit",
    "cahier":           "petit",
    "stylo":            "petit",
    "cle":              "petit",
    "telecommande":     "petit",
    "souris":           "petit",
    "bougie":           "petit",
    "savon":            "petit",
    "peluche":          "petit",
    "balle":            "petit",
    "cube":             "petit",
    "photo":            "petit",

    # equivalents anglais (config lang=en)
    "washing_machine":  "grand",
    "refrigerator":     "grand",
    "dishwasher":       "grand",
    "sofa":             "grand",
    "wardrobe":         "grand",
    "bookshelf":        "grand",
    "desk":             "grand",
    "chair":            "moyen",
    "lamp":             "moyen",
    "plant":            "moyen",
    "laptop":           "moyen",
    "bottle":           "moyen",
    "cup":              "petit",
    "plate":            "petit",
    "apple":            "petit",
    "banana":           "petit",
    "book":             "petit",
    "pen":              "petit",
    "key":              "petit",
    "remote":           "petit",
}

# noms francais pour les IDs anglais (utilises dans les prompts et la validation hors config "anglais")
OBJECT_NAMES_FR = {
    "washing_machine": "lave-linge",
    "refrigerator":    "frigo",
    "dishwasher":      "lave-vaisselle",
    "sofa":            "canape",
    "wardrobe":        "armoire",
    "bookshelf":       "bibliotheque",
    "desk":            "bureau",
    "chair":           "chaise",
    "lamp":            "lampe",
    "plant":           "plante",
    "laptop":          "ordinateur",
    "bottle":          "bouteille",
    "cup":             "tasse",
    "plate":           "assiette",
    "apple":           "pomme",
    "banana":          "banane",
    "book":            "livre",
    "pen":             "stylo",
    "key":             "cle",
    "remote":          "telecommande",
}

TAILLE_SCORE = {"grand": 3, "moyen": 2, "petit": 1}

RELATION_TYPES = [
    "on",
    "under",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "against",
    "inside",
]

# relations qui supportent un champ "distance" (en metres)
RELATION_TYPES_WITH_DISTANCE = {"left_of", "right_of", "in_front_of", "behind"}

# orientations valides (coherent avec placement_v1_distances_orientations.py)
ORIENTATIONS = [
    "tip_left",
    "tip_right",
    "tip_forward",
    "tip_backward",
    "upside_down",
    "turn_left",
    "turn_right",
    "turn_around",
]

# fourchettes (= plages min/max) de distance realistes selon la taille de l'objet
# le plus grand de la paire
DISTANCE_RANGES_BY_SIZE = {
    "petit": (0.05, 0.5),
    "moyen": (0.2, 1.5),
    "grand": (0.5, 3.0),
}

# pas d'arrondi pour les distances  
DISTANCE_ROUND = 0.05

# probabilites par defaut (s'appliquent a toutes les configs)
DEFAULT_DISTANCE_PROB    = 0.25  # par relation eligible
DEFAULT_ORIENTATION_PROB = 0.15  # par objet

STYLE_EXAMPLES = [
    "je veux un frigo a cote d'une machine a laver",
    "une table avec deux chaises devant",
    "je veux un bureau avec une lampe dessus et une chaise devant",
    "un lit contre le mur avec une table de nuit a sa droite",
    "place le micro onde sur le frigo",
    "je voudrais un frigo avec a sa gauche un lave linge a 50 cm",
    "un mug a l envers sur une table",
    "une banane couchee a 30 cm d un mug",
    "une plante a 1m5 derriere un canape, avec une lampe renversee a sa droite",
    "un toaster tourne vers la gauche a cote du four",
]
