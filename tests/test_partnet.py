import os
import genesis as gs

OBJETS_DIR = os.path.join(os.path.dirname(__file__), '..', 'objets')

def test_object(filename):
    path = os.path.join(OBJETS_DIR, filename)
    print(f"\n=== {filename} ===")
    print(f"path: {path}")

    gs.init(backend=gs.cpu)
    scene = gs.Scene(show_viewer=True, sim_options=gs.options.SimOptions(dt=0.01))
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.URDF(file=path),
        material=gs.materials.Rigid(rho=1000),
    )
    scene.build()
    while scene.viewer.is_alive():
        scene.step()
    gs.destroy()


def main():
    liste_partnet_marche = [
        "154_robinet/mobility.urdf",
        "8867_porte_battante_2/mobility.urdf",
        "9032_porte_coulissante/mobility.urdf",
        "10357_poubelle/mobility.urdf",
        "100073_cle_usb/mobility.urdf",
    ]

    liste_partnet_marche_pas = [
        "10143_refrigerateur/mobility.urdf",
        "10211_ordinateur_portable/mobility.urdf",
        "11826_lave_linge/mobility.urdf",
        "44817_meuble_tiroirs/mobility.urdf",
        "100658_boite_carton_ouverte/mobility.urdf",
        "103369_lave_vaisselle/mobility.urdf",
        "103477_toaster/mobility.urdf",
    ]

    liste_partnet = [
        "154_robinet/mobility.urdf",
        "7138_four/mobility.urdf",
        "8867_porte_battante_2/mobility.urdf",
        "9032_porte_coulissante/mobility.urdf",
        "10143_refrigerateur/mobility.urdf",
        "10211_ordinateur_portable/mobility.urdf",
        "10357_poubelle/mobility.urdf",
        "11826_lave_linge/mobility.urdf",
        "44817_meuble_tiroirs/mobility.urdf",
        "100073_cle_usb/mobility.urdf",
        "100658_boite_carton_ouverte/mobility.urdf",
        "103369_lave_vaisselle/mobility.urdf",
        "103477_toaster/mobility.urdf",
    ]

    liste_ycb = [
        "011_banane/google_16k/model.urdf",
        "035_power_drill/google_16k/model.urdf",
        "037_scissors/google_16k/model.urdf",
    ]

    # --- choix de la liste a tester ---
    liste = liste_partnet_marche_pas

    total = len(liste)
    for i, filename in enumerate(liste):
        print(f"\n[{i+1}/{total}] Ferme la fenetre pour passer au suivant.")
        try:
            test_object(filename)
        except Exception as e:
            print(f"ERREUR sur {filename}: {e}")

    print("\nTous les objets ont ete testes.")


if __name__ == "__main__":
    main()
