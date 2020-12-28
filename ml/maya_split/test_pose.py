import os
import csv
import itertools
import maya.cmds as mc
import maya.OpenMaya as OpenMaya
import maya.api.OpenMaya as om


# the folder containing rigged parameters
input_path = "D:/ACG/project/ml/maya_split/gen_data/mover_rigged/"
temp_path = "D:/ACG/project/ml/maya_split/gen_data/temp_data/"
out_path = "D:/ACG/project/ml/maya_split/gen_data/data_set/"
test_path = "D:/ACG/project/ml/maya_split/gen_data/test_data/"
recon_path = "D:/ACG/project/ml/maya_split/gen_data/recon/"

topology_path = "D:/ACG/project/ml/maya_split/gen_data/topology_Mery_geo_cn_body.csv"
anchor_path = "D:/ACG/project/ml/maya_split/gen_data/face_anchor.csv"
joint_path = "D:/ACG/project/ml/maya_split/gen_data/joint_relation.csv"

# filename = "rigged0.csv"

# Macro definition
TEMP_BS_NODE = 'mlBlendShape'
TEMP_TARGET = 'mlTarget'
SKIN_TYPES = ['skinCluster']

MESH = "Mery_geo_cn_body"
PRECISION = 8


def prep_mesh(mesh):
    # get the deformers before operations
    mc.deformer(mesh, frontOfChain=True, type='blendShape', name=TEMP_BS_NODE)
    history = mc.listHistory(mesh, interestLevel=1)
    history = [i for i in history if 'geometryFilter' in mc.nodeType(i, inherited=True)]

    deformers = []
    for his in history:
        if his == TEMP_BS_NODE:
            break
        deformers.append(his)
    return deformers


def read_in_data(filepath):
    # get the data files to read
    files = []
    for root, dirs, fs in os.walk(filepath):
        files = fs
    # read in the result
    pose_data = {}
    for file_name in files:
        # for each file, read in as a dict
        f = open(filepath + '/' + file_name, 'r')
        temp = {}
        reader = csv.reader(f)
        content = list(reader)
        if file_name[:-4] in ['anchorPoints', 'differentialOffset', 'localOffset', 'worldOffset', 'worldPos']:
            for line in content:
                temp[int(line[0])] = [float(coord) for coord in line[1:]]
        elif file_name[:-4] in ['jointLocalMatrix', 'jointLocalQuaternion', 'jointWorldMatrix', 'jointWorldQuaternion']:
            for line in content:
                temp[line[0]] = [float(coord) for coord in line[1:]]
        else:
            for line in content:
                temp[line[0]] = line[1:]
        f.close()
        pose_data[file_name[:-4]] = temp

    return pose_data


def read_in_rig(filepath):
    # read in the result as a dict
    f = open(filepath, 'r')
    reader = list(csv.reader(f))
    pose_rig = {}
    # get the attribute to set
    attribute = reader[0][1:]
    for line in reader[1:]:
        mover = line[0]
        # get each attribute
        for i, value in enumerate(line[1:]):
            pose_rig[mover + '.' + attribute[i]] = float(value)
    f.close()
    return pose_rig


def getEmptyShapeData():
    return {'shapes': {}, 'setMembers': True, 'baseWeights': {}, 'nodeType': 'blendShape'}


def getEmptyShape():
    return {'offsets': {}, 'weights': {}, 'shapeIndex': None}


def getShapeAliasLookup(node):
    """
    Builds a lookup dictionary that maps a blendShape node's weight
    attribute alias name to the weight attribute's index.
    """
    aliasLookup = {}
    weightIndices = mc.getAttr(node + ".weight", mi=True)
    if weightIndices:
        for weightIndex in weightIndices:
            attributeAlias = mc.aliasAttr(node + ".weight[" + str(weightIndex) + "]", q=True)
            if attributeAlias:
                aliasLookup[attributeAlias] = weightIndex
            else:
                aliasLookup['[' + str(weightIndex) + ']'] = weightIndex

    return aliasLookup


def setShapeData(node,
                 shapeData,
                 inputIndex=0,
                 shapes=None):
    """
    sets the shape data onto a blendShape node.
    :param str node: the blendShape to add the shapes to.
    :param dict shapeData: the shape data to apply to the node.
    :param list shapes: if given, only modify given target names
    """
    targetName = None
    if shapes:
        targetName = shapes[0]
    if not mc.objExists('%s.%s' % (node, targetName)):
        shapeIndex = 0
        weightIndices = mc.getAttr(node + ".weight", multiIndices=True)
        if weightIndices:
            shapeIndex = weightIndices[-1] + 1
        attr = '%s.weight[%i]' % (node, shapeIndex)
        mc.getAttr(attr)
        mc.aliasAttr(targetName, attr)
        mc.setAttr(attr, 1.0)

    nodeType = mc.nodeType(node)
    inputShape = mc.deformer(node, g=True, q=True)
    shapeAliasLookup = getShapeAliasLookup(node)
    print(shapeAliasLookup)

    if not 'shapes' in shapeData:
        print("procedureName" + ':  shapeData does not have a "shapes" key.  Returning now...')
        return

    for shapeAlias in shapeData['shapes']:
        if shapes and shapeAlias not in shapes:
            continue
        print(shapeAlias)

        # read the information stored for this shape
        targetData = shapeData['shapes'][shapeAlias]
        targetOffsets = targetData["offsets"]
        targetWeights = targetData["weights"]
        shapeIndex = shapeAliasLookup.get(shapeAlias, None)
        print('shapeIndex: ' + str(shapeIndex))

        # iterate through the offset dictionary
        pointList = []
        componentList = []

        for pointIndex in targetOffsets:
            pointData = targetOffsets[pointIndex]
            pointList.append((pointData[0], pointData[1], pointData[2], 1.0))
            componentList.append('vtx[' + str(pointIndex) + ']')

        # create the element by calling getAttr
        try:
            mc.getAttr(node + '.inputTarget[' + str(inputIndex) + ']')
        except:
            pass
        try:
            mc.getAttr(node + '.inputTarget[' + str(inputIndex) + '].inputTargetGroup[' + str(shapeIndex) + ']')
        except:
            pass

        shapeAttr = node + ".inputTarget[" + str(inputIndex) + "].inputTargetGroup[" + str(shapeIndex) + "]"
        mc.setAttr(shapeAttr + ".inputTargetItem[6000].inputPointsTarget", len(componentList), type="pointArray", *pointList)
        mc.setAttr(shapeAttr + ".inputTargetItem[6000].inputComponentsTarget", len(componentList), type="componentList", *componentList)

        tAttrs = mc.listAttr(shapeAttr, m=True, string='targetWeights')
        if tAttrs != None:
            for a in tAttrs:
                mc.removeMultiInstance((node + '.' + a), b=True)
        # set the weights
        for weight in targetWeights:
            mc.setAttr(shapeAttr + ".targetWeights[" + str(weight) + "]", targetWeights[weight])



def reconstruction(rigpath, datapath, training_type='local_offset'):
    # Create deformers
    deformers = prep_mesh(MESH)
    deformer_env_dict = {}
    # shutdown all deformers except for skin clusters
    for deformer in deformers:
        dtype = mc.nodeType(deformer)
        if dtype not in SKIN_TYPES:
            deformer_env_dict[deformer] = mc.getAttr(deformer + '.envelope')
            mc.setAttr(deformer + '.envelope', 0.0)
    # set the deformers
    BLENDSHAPE = TEMP_BS_NODE
    mesh = mc.deformer(BLENDSHAPE, query=True, geometry=True)[0]
    print(mesh)

    # read in data for reconstruction
    pose_rig = read_in_rig(rigpath)
    pose_data = read_in_data(datapath)

    # pose the rig first
    for mover in pose_rig:
        if mc.getAttr(mover, l=True):
            continue
        mc.setAttr(mover, float(pose_rig[mover]))

    # get shape data
    shape_data = getEmptyShapeData()
    shape_data['shapes'][TEMP_TARGET] = getEmptyShape()

    # reconstruct the local offset type
    if training_type == 'local_offset':
        cur_offset_data = pose_data['localOffset']
        for vtx_id in cur_offset_data:
            shape_data['shapes'][TEMP_TARGET]['offsets'][vtx_id] = cur_offset_data[vtx_id]
    elif training_type == 'differntial':
        cur_offset_data = pose_data['differentialOffset']
        for vtx_id in cur_offset_data:
            shape_data['shapes'][TEMP_TARGET]['offsets'][vtx_id] = cur_offset_data[vtx_id]

    setShapeData(BLENDSHAPE, shape_data, shapes=[TEMP_TARGET])
    mc.setAttr(BLENDSHAPE + '.' + TEMP_TARGET, 1.0)


