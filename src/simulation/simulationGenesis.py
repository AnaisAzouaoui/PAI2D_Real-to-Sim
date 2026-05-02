import os
import tempfile
import xml.etree.ElementTree as ET
import trimesh
import genesis as gs


def patch_urdf(urdf_path):
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall('link'):
        # collecter les mesh existants pour ce link
        existing_meshes = []
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
                    if tag == 'visual':
                        existing_meshes.append(full)

        # calculer bbox et remplacer collision par box 
        if existing_meshes:
            try:
                combined = trimesh.util.concatenate([
                    trimesh.load(p, force='mesh') for p in existing_meshes
                ])
                w, d, h = combined.bounding_box.extents
                mass = max(0.01, 500.0 * w * d * h)
                ixx = mass / 12.0 * (d**2 + h**2)
                iyy = mass / 12.0 * (w**2 + h**2)
                izz = mass / 12.0 * (w**2 + d**2)
                box_size = f'{w:.6f} {d:.6f} {h:.6f}'
            except Exception:
                mass, ixx, iyy, izz = 0.1, 0.001, 0.001, 0.001
                box_size = '0.1 0.1 0.1'
        else:
            mass, ixx, iyy, izz = 0.001, 1e-6, 1e-6, 1e-6
            box_size = '0.01 0.01 0.01'

        # remplacer toutes les collisions par une box
        for col in link.findall('collision'):
            link.remove(col)
        col_elem = ET.SubElement(link, 'collision')
        ET.SubElement(ET.SubElement(col_elem, 'geometry'), 'box', size=box_size)

        # remplacer l'inertial
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


def mesh_to_urdf(mesh_path, density=500.0):
    """Genere un URDF avec collision en box et visuel en mesh"""
    mesh_path = os.path.abspath(mesh_path)
    m = trimesh.load(mesh_path, force='mesh')
    w, d, h = m.bounding_box.extents
    mass = max(0.01, density * w * d * h)
    ixx = mass / 12.0 * (d**2 + h**2)
    iyy = mass / 12.0 * (w**2 + h**2)
    izz = mass / 12.0 * (w**2 + d**2)

    urdf_content = f"""<?xml version="1.0"?>
<robot name="mesh_obj">
  <link name="base">
    <visual>
      <geometry><mesh filename="{mesh_path}"/></geometry>
    </visual>
    <collision>
      <geometry><box size="{w:.6f} {d:.6f} {h:.6f}"/></geometry>
    </collision>
    <inertial>
      <mass value="{mass:.6f}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="{ixx:.6f}" iyy="{iyy:.6f}" izz="{izz:.6f}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
</robot>"""

    tmp = tempfile.NamedTemporaryFile(suffix='.urdf', delete=False, mode='w', encoding='utf-8')
    tmp.write(urdf_content)
    tmp.close()
    return tmp.name


def create_scene(objetsList):
    '''
    Cree la scene Genesis a partir de la liste d'objets.
    '''
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        show_viewer=True,
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

        if path.endswith('.urdf'):
            path = patch_urdf(path)
        elif path.endswith('.xml'):
            xml_dir = os.path.dirname(path)
            found = None
            for name in ["textured.obj", "nontextured.stl", "nontextured.ply"]:
                candidate = os.path.join(xml_dir, name)
                if os.path.exists(candidate):
                    found = candidate
                    break
            if found is None:
                raise FileNotFoundError(f"Aucun mesh trouve a cote de {path}")
            path = mesh_to_urdf(found)
        else:
            path = mesh_to_urdf(path)

        morph = gs.morphs.URDF(file=path, pos=pos, quat=quat, scale=scale, fixed=False)
        scene.add_entity(morph, material=gs.materials.Rigid(rho=1000, friction=0.5))

    scene.build()

    while True:
        scene.step()
