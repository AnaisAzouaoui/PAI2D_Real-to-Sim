import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from app import MainWindow
from promptToJson_ import object_dim_quat, object_rec

OBJETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "objets")

# ca c'est ceux a quoi va ressembler un objet list que mon truc va rendre 
OBJETS_LIST_TEST = [
    {
        "id": "lave_linge",
        "urdf": "11826_lave_linge",
        "path": os.path.join(OBJETS_DIR, "11826_lave_linge", "mobility.urdf"),
        "scale": 0.55,
        "quat": (0.7071, 0.0, 0.0, 0.7071),
        "pos": (0.0, 0.0, 0.43),
    },
    {
        "id": "poubelle",
        "urdf": "10357_poubelle",
        "path": os.path.join(OBJETS_DIR, "10357_poubelle", "mobility.urdf"),
        "scale": 0.26,
        "quat": (0.7071, 0.0, 0.0, 0.7071),
        "pos": (0.0, 0.0, 1.07),
    },
]

user_prompt = "place une poubelle sur un lave linge"
objets_reconnus, non_reconnus = object_rec(user_prompt)
print("Reconnus:", objets_reconnus)
print("Non reconnus:", non_reconnus)
OBJETS_LIST_TEST_2 = object_dim_quat(user_prompt,objets_reconnus)
print(OBJETS_LIST_TEST_2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    def load_test_scene():
        try:
            for obj in OBJETS_LIST_TEST_2:
                print(json.dumps(obj, indent=2))
            window._on_scene_ready(OBJETS_LIST_TEST_2)
            window._append_message(
                "TEST Scene hardcodee: lave-linge + poubelle a cote",
                "system"
            )
        except Exception as e:
            window._append_message(f"[TEST] Erreur : {e}", "error")

    QTimer.singleShot(0, load_test_scene)
    sys.exit(app.exec())
