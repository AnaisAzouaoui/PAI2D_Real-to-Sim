import os
import platform
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import genesis as gs


def collect_visual_meshes(link, urdf_dir):
    meshes = []
    for block in link.findall('visual'):
        geom = block.find('geometry')
        if geom is None:
            continue
        mesh_tag = geom.find('mesh')
        if mesh_tag is None:
            continue
        full = mesh_tag.get('filename', '')
        if not os.path.exists(full):
            continue
        try:
            m = trimesh.load(full, force='mesh')
            origin = block.find('origin')
            if origin is not None:
                T = np.eye(4)
                T[:3, :3] = Rotation.from_euler('xyz', [float(v) for v in origin.get('rpy', '0 0 0').split()]).as_matrix()
                T[:3, 3] = [float(v) for v in origin.get('xyz', '0 0 0').split()]
                m.apply_transform(T)
            meshes.append(m)
        except Exception:
            pass
    return meshes


def patch_urdf(urdf_path):
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall('link'):
        # Mettre a jour les chemins absolus et retirer les refs de fichiers manquants
        for tag in ('visual', 'collision'):
            for block in link.findall(tag):
                geom = block.find('geometry')
                if geom is None:
                    continue
                mesh_tag = geom.find('mesh')
                if mesh_tag is None:
                    continue
                filename = mesh_tag.get('filename', '')
                full = os.path.join(urdf_dir, filename)
                if not os.path.exists(full):
                    print(f"[patch_urdf] mesh absent retire : {filename}")
                    link.remove(block)
                else:
                    mesh_tag.set('filename', full)

        # Si le link n'a aucune collision apres nettoyage: ajouter une box de fallback
        # centree sur le vrai mesh (avec les offsets URDF appliques)
        if not link.findall('collision'):
            meshes = collect_visual_meshes(link, urdf_dir)
            if meshes:
                try:
                    combined = trimesh.util.concatenate(meshes)
                    w, d, h = combined.bounding_box.extents
                    cx, cy, cz = combined.bounding_box.centroid
                    box_size = f'{w:.6f} {d:.6f} {h:.6f}'
                    box_origin = f'{cx:.6f} {cy:.6f} {cz:.6f}'
                except Exception:
                    box_size = '0.1 0.1 0.1'
                    box_origin = '0 0 0'
            else:
                box_size = '0.01 0.01 0.01'
                box_origin = '0 0 0'
            col_elem = ET.SubElement(link, 'collision')
            ET.SubElement(col_elem, 'origin', xyz=box_origin, rpy='0 0 0')
            ET.SubElement(ET.SubElement(col_elem, 'geometry'), 'box', size=box_size)

        # Recalculer l'inertial a partir des meshes visuels avec leurs origins
        meshes = collect_visual_meshes(link, urdf_dir)
        if meshes:
            try:
                combined = trimesh.util.concatenate(meshes)
                w, d, h = combined.bounding_box.extents
                mass = max(0.01, 500.0 * w * d * h)
                ixx = mass / 12.0 * (d**2 + h**2)
                iyy = mass / 12.0 * (w**2 + h**2)
                izz = mass / 12.0 * (w**2 + d**2)
            except Exception:
                mass, ixx, iyy, izz = 0.1, 0.001, 0.001, 0.001
        else:
            mass, ixx, iyy, izz = 0.001, 1e-6, 1e-6, 1e-6

        old = link.find('inertial')
        if old is not None:
            link.remove(old)
        inertial = ET.SubElement(link, 'inertial')
        ET.SubElement(inertial, 'mass', value=f'{mass:.6f}')
        ET.SubElement(inertial, 'origin', xyz='0 0 0', rpy='0 0 0')
        ET.SubElement(inertial, 'inertia',
            ixx=f'{ixx:.8f}', iyy=f'{iyy:.8f}', izz=f'{izz:.8f}',
            ixy='0', ixz='0', iyz='0')

    tmp = tempfile.NamedTemporaryFile(suffix='.urdf', delete=False, mode='w', encoding='utf-8')
    tree.write(tmp.name, encoding='unicode', xml_declaration=True)
    tmp.close()
    return tmp.name


def make_morph(path, **kwargs):
    """Cree le bon morph Genesis selon le type de fichier"""
    if path.endswith('.xml'):
        xml_dir = os.path.dirname(path)
        for name in ["textured.obj", "nontextured.stl", "nontextured.ply"]:
            candidate = os.path.join(xml_dir, name)
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            raise FileNotFoundError(f"Aucun mesh trouve a cote de {path}")
    if path.endswith(('.urdf', '.xacro')):
        return gs.morphs.URDF(file=patch_urdf(path), **kwargs)
    return gs.morphs.Mesh(file=path, **kwargs)


def create_scene(objetsList):
    '''
    Cree la scene Genesis a partir de la liste d'objets.
    '''
    gs.init(backend=gs.cpu)

    # Camera positionnee au meme angle qu'azimuth=225 dans pyqtgraph
    # pour que left/right corresponde a ce qu'on voit dans la 3D view
    cam_kwargs = dict(
        camera_pos=(-3.5, 3.5, 2.5),
        camera_lookat=(0.0, 0.0, 0.8),
        camera_fov=40,
    )
    viewer_options = gs.options.ViewerOptions(run_in_thread=False, **cam_kwargs) if platform.system() == "Darwin" else gs.options.ViewerOptions(**cam_kwargs)

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=viewer_options,
        rigid_options=gs.options.RigidOptions(
            dt=0.001,
            integrator=gs.integrator.Euler,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            show_link_frame=True,
        )
    )
    scene.add_entity(gs.morphs.Plane())

    for obj in objetsList:
        path  = obj['path']
        pos   = obj['pos']
        quat  = obj.get('quat', [0.0, 0.0, 0.0, 1.0])
        scale = obj.get('scale', 1.0)

        scene.add_entity(
            make_morph(path, pos=pos, quat=quat, scale=scale, fixed=False),
            material=gs.materials.Rigid(rho=1000, friction=0.5)
        )

    scene.build()

    while True:
        scene.step()
