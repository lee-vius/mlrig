import maya.cmds as mc
import maya.api.OpenMaya as om
import csv

folder_path = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/"

def get_connection_info(mesh):
    # get the connection relation of the given mesh
    connection_map = {}
    selection = om.MSelectionList()
    if mc.nodeType(mesh) == 'transform':
        mesh = mc.listRelatives(mesh, children=True)[0]
    selection.add(mesh)
    geom = selection.getDependNode(0)
    vtx_iter = om.MItMeshVertex(geom)
    while not vtx_iter.isDone():
        index = vtx_iter.index()
        connected_vtx = vtx_iter.getConnectedVertices()
        connection_map[index] = connected_vtx
        vtx_iter.next()

    return connection_map
    

def gen_connection_file(mesh_list, filename='topology_'):
    # generate the file containing connection info for a mesh
    for mesh in mesh_list:
        # get the connection info
        connection_map = get_connection_info(mesh)
        f = open(folder_path + filename + mesh + '.csv', 'w')
        csv_writer = csv.writer(f)
        for _, key in enumerate(connection_map):
            csv_writer.writerow([key] + list(connection_map[key]))
        f.close()


