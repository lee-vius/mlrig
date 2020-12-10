import os
import csv
import itertools
import maya.cmds as mc
import maya.OpenMaya as OpenMaya
import maya.api.OpenMaya as om

import numpy
# from util.py import *

# the folder containing rigged parameters
input_path = "D:/ACG/project/ml/maya_split/gen_data/mover_rigged/"
temp_path = "D:/ACG/project/ml/maya_split/gen_data/temp_data/"
out_path = "D:/ACG/project/ml/maya_split/gen_data/data_set/"

topology_path = "D:/ACG/project/ml/maya_split/gen_data/topology_Mery_geo_cn_body.csv"
anchor_path = "D:/ACG/project/ml/maya_split/gen_data/face_anchor.csv"
joint_path = "D:/ACG/project/ml/maya_split/gen_data/joint_relation.csv"

filename = "rigged0.csv"

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


def get_rest_mesh(mesh, precision):
    # the file containing rest information of the mesh
    rest_file_path = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/rest_mesh.csv"
    # get world positions of rest points
    _, worldPos = get_worldPos(mesh, precision)
    f = open(rest_file_path, 'w')
    csv_writer = csv.writer(f)
    for i, _ in enumerate(worldPos):
        csv_writer.writerow([i] + list(worldPos[i]))

    f.close()


def getPositions(obj):
    # get the mesh points positions
    selectionList = om.MSelectionList()
    selectionList.add(obj)
    geomObj = selectionList.getDependNode(0)
    pos = None
    geomIt = om.MFnMesh(geomObj)
    pos = geomIt.getPoints()
    return pos


def get_worldPos(mesh, precision):
    # get world position of given mesh
    meshShape = mc.listRelatives(mesh, children=True)
    meshShape = mc.ls(meshShape, ni=True)
    meshShape = meshShape[0]
    positions = getPositions(meshShape)
    # generate output dictionary
    worldPos = {}
    for i, p in enumerate(positions):
        worldPos[i] = [round(p[0], precision), round(p[1], precision), round(p[2], precision)]
    
    return meshShape, positions, worldPos


def get_worldOffset(mesh, precision, meshShape, ori_pos):
    # get world offset of given mesh
    # NOTE: only use if after get_worldPos
    linear_pos = getPositions(meshShape)
    # generate output dictionary
    worldOffset = {}
    for i, p in enumerate(linear_pos):
        offset = ori_pos[i] - p
        worldOffset[i] = [
            round(offset[0], precision), 
            round(offset[1], precision), 
            round(offset[2], precision)
        ]

    return linear_pos, worldOffset


def get_localOffset(resultMesh, sculptMesh, blendShapeNode=TEMP_BS_NODE, targetName=TEMP_TARGET, tol=0.0001):
    # calculate offset in local space before linear skin blending
    # resultMesh: mesh object only containing linear skin blending
    # sculptMesh: the deformed mesh resulted from a full rig with all deformers
    if mc.nodeType(resultMesh) == 'transform':
        resultMesh = mc.listRelatives(resultMesh, children=True, path=True)
        resultMesh = mc.ls(resultMesh, noIntermediate=True)[0]

    if mc.nodeType(sculptMesh) == 'transform':
        sculptMesh = mc.listRelatives(sculptMesh, children=True, path=True)
        sculptMesh = mc.ls(sculptMesh, noIntermediate=True)[0]

    # get result and sculpt positions
    # and get the offset vectors between the two
    targetWeight = None
    weightAttr = '%s.%s' % (blendShapeNode, targetName)
    if mc.objExists(weightAttr):
        targetWeight = mc.getAttr(weightAttr)
        mc.setAttr(weightAttr, 0.0)
    resultPos = getPositions(resultMesh)
    sculptPos = getPositions(sculptMesh)

    if targetWeight:
        mc.setAttr(weightAttr, targetWeight)
    if len(resultPos) != len(sculptPos):
        # number of points doesn't match -- raise an exception
        raise RuntimeError("The number of result components does not match")

    # This offset is got directly from difference between world pos and linear pos
    offsetVec = [e[0] - e[1] for e in zip(sculptPos, resultPos)] 

    # continue to get local offset
    prunedResultPos = []
    prunedResultInd = []
    prunedOffsetVec = []

    for index, vec in enumerate(offsetVec):
        if vec.length() <= tol:
            continue
        prunedResultPos.append(resultPos[index])
        prunedResultInd.append(index)
        prunedOffsetVec.append(vec)
    
    if not prunedResultInd:
        print("makeCorrectiveFromeSculpt: Source and Destination geometries are in sync ... No action takes!")
        return {}
    
    sData = makeLinearCorrective(prunedOffsetVec, blendShapeNode, targetName, deformedGeometry=resultMesh, resultPos=prunedResultPos, resultInd=prunedResultInd)
    return sData


def makeLinearCorrective(offsetVec, blendShapeNode, targetName, deformedGeometry=None, resultPos=None, resultInd=None):
    # make linear corrective on the computed offset
    overrideTargetName = None
    shapeData = None

    matricesResult = genMatrix(blendShapeNode, deformedGeometry=deformedGeometry, resultPos=resultPos, resultInd=resultInd, targetName=overrideTargetName)

    invShapeMatrices = matricesResult['invShapeMatrices']
    nonInvertablePointsSet = matricesResult['nonInvertablePointsSet']

    # remove any invalid inverseMatrices from consideration
    offsetVecToUse = []
    invShapeMatricesToUse = {}
    resultIndToUse = []
    for index, ov in itertools.izip(resultInd, offsetVec):
        if index in nonInvertablePointsSet:
            continue
        invShapeMatricesToUse[index] = invShapeMatrices[index]
        offsetVecToUse.append(ov)
        resultIndToUse.append(index)

    optionalData = {}
    optionalData[targetName] = dict(zip(resultIndToUse, offsetVecToUse))

    applyMatricesResult = applyMatricesToTransformBlendShapeNodeOffsets(
        blendShapeNode,
        invShapeMatricesToUse,
        shapeData=shapeData,
        targets=[targetName],
        optionalData=optionalData,
        matrixOp=matrixOp_makeLinearCorrective
    )

    indicesToUse = applyMatricesResult['matrixOpResults'][targetName]
    deltaOffsetsModificationsValues = [applyMatricesResult['matrixOpResults'][targetName][x]['multResult'] for x in indicesToUse]
    deltaOffsetsModifications = dict(zip(indicesToUse, deltaOffsetsModificationsValues))

    return deltaOffsetsModifications 


def matrixOp_makeLinearCorrective(currentMatrix, cv, optionalData=None):
    ov = optionalData
    multResult = None
    addResult = None
    
    if (ov):
        multResult = numpy.dot(currentMatrix, numpy.append(ov, 0))
        addResult = multResult + numpy.append(cv, 0)
    else:
        # pass-through if ov is not supplied
        multResult = None
        addResult = cv

    result = {'multResult': multResult[:-1], 'addResult': addResult[:-1]}
    # the goal is to modify the cv by the offsetValue modified by the matrix: so the add Result is the shapeOffsetReplacement result
    result['shapeOffsetReplacement'] = addResult
    return result


def applyMatricesToTransformBlendShapeNodeOffsets(blendShapeNode, matrices, shapeData=None, targets=None, optionalData=None, matrixOp=matrixOp_makeLinearCorrective):
    applyShapeData = getEmptyShapeData()
    for target in targets:
       shape = getEmptyShape()
       applyShapeData['shapes'][target] = shape

    applyMatricesResult = applyMatricesToTransformShapeDataOffsets(
        applyShapeData,
        matrices,
        targets=targets,
        optionalData=optionalData,
        matrixOp=matrixOp
    )

    return applyMatricesResult


def applyMatricesToTransformShapeDataOffsets(shapeData, matrices, targets=None, matrixOp=matrixOp_makeLinearCorrective, optionalData=None):
    targetsToUse = None
    if targets:
        targetsSet = set(targets)
        shapeDataTargetsSet = set(shapeData['shapes'])
        commonTargetsSet = targetsSet & shapeDataTargetsSet
        targetsToUse = list(commonTargetsSet)
    else:
        targetsToUse = shapeData['shapes']

    matrixOpResults = {}
    errorPoints = {}
    shapeOffsetsAll = {}
    for currentShape in targetsToUse:
        shapeOffsetsAll[currentShape] = {}
        matrixOpResults[currentShape] = {}
        
        shapeOffsets = shapeData['shapes'][currentShape]['offsets']

        for index, currentMatrix in matrices.iteritems():
            cv = shapeOffsets.get(index, (0, 0, 0))
            if (optionalData):
                currentOptionalData = optionalData[currentShape][index]
            else:
                currentOptionalData = None
            
            currentMatrixOpResult = matrixOp(currentMatrix, cv, optionalData=currentOptionalData)
            matrixOpResults[currentShape][index] = currentMatrixOpResult
            # modify the current shapeOffset with the replacement value from the matrixOp
            shapeOffsets[index] = currentMatrixOpResult['shapeOffsetReplacement']
        shapeOffsetsAll[currentShape] = shapeOffsets
    result = {}
    result['shapeOffsetsAll'] = shapeOffsetsAll
    result['errorPoints'] = errorPoints
    result['matrixOpResults'] = matrixOpResults
    result['targetsToUse'] = targetsToUse

    return result


def genMatrix(blendShapeNode, deformedGeometry=None, resultPos=None, resultInd=None, targetName=None):
    # The procedure is meant to return transform matrices which describe how a 
    # 1-unit offset in each X, Y, Z direction actually transforms to an offset
    # at the end of the subsequent deformer stack
    removeTarget = False
    if not targetName:
        targetName = 'CORRECTIVE_DUMMY_TARGET'
        removeTarget = True
    # initialize empty shape data
    shapeData = getEmptyShapeData()
    shape = getEmptyShape()
    shapeData['shapes'][targetName] = shape
    shape['offsets'] = {}

    if not mc.objExists('%s.%s' % (blendShapeNode, targetName)):
        shapeIndex = 0
        weightIndices = mc.getAttr(blendShapeNode + ".weight", multiIndices=True)
        if weightIndices:
            shapeIndex = weightIndices[-1] + 1
        attr = '%s.weight[%i]' % (blendShapeNode, shapeIndex)
        mc.getAttr(attr)
        mc.aliasAttr(targetName, attr)
        mc.setAttr(attr, 1.0)

    # get the x, y, z basis vectors for each point
    perAxisUnitOffsetVectors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    axes = []

    for ii in range(3):
        for ind in resultInd:
            shape['offsets'][ind] = perAxisUnitOffsetVectors[ii]

        setBlendShapeData(blendShapeNode, shapeData)
        currentAxisOffsetPos = getPositions(deformedGeometry)

        newCurrentAxisOffsetPos = [currentAxisOffsetPos[j] for j in resultInd]
        currentAxes = [e[0] - e[1] for e in zip(newCurrentAxisOffsetPos, resultPos)]
        axes.append(currentAxes)

    if removeTarget:
        mc.aliasAttr(blendShapeNode + '.' + targetName, remove=True)
        mc.removeMultiInstance(blendShapeNode + '.weight[' + str(shapeIndex) + ']', b=True)
        mc.removeMultiInstance(blendShapeNode + '.inputTarget[0].inputTargetGroup[' + str(shapeIndex) + '].inputTargetItem[6000]')
        mc.removeMultiInstance(blendShapeNode + '.inputTarget[0].inputTargetGroup[' + str(shapeIndex) + ']')

    xAxes = axes[0]
    yAxes = axes[1]
    zAxes = axes[2]

    nonInvertablePoints = []
    nonInvertablePointSet = set()

    # calculate the shapeMatrix first
    matrices = {}
    for index, xAxis, yAxis, zAxis in itertools.izip(resultInd, xAxes, yAxes, zAxes):
        shapeMatrix = numpy.array([
            [xAxis[0], xAxis[1], xAxis[2], 0],
            [yAxis[0], yAxis[1], yAxis[2], 0],
            [zAxis[0], zAxis[1], zAxis[2], 0],
            [0, 0, 0, 1]
        ])
        matrices[index] = shapeMatrix
    
    result = {}
    result['matrices'] = matrices

    invShapeMatrices = {}
    # calculate the invShapeMatrix first
    invShapeMatrices, nonInvertablePointSet, nonInvertablePoints = invertMatrices(matrices)

    result['invShapeMatrices'] = invShapeMatrices
    result['nonInvertablePoints'] = nonInvertablePoints
    result['nonInvertablePointsSet'] = nonInvertablePointSet
    return result


def invertMatrices(matrices):
    invMatrices = {}
    nonInvertablePoints = []
    nonInvertablePointsSet = set()

    for ii in matrices:
        invMatrix = None
        try:
            invMatrix = numpy.linalg.inv(matrices[ii])
        except:
            nonInvertablePoints.append(str(ii))
            nonInvertablePointsSet.add(ii)
        else:
            invMatrices[ii] = invMatrix

    return invMatrices, nonInvertablePointsSet, nonInvertablePoints


def getEmptyShapeData():
    return {'shapes': {}, 'setMembers': True, 'baseWeights': {}, 'nodeType': 'blendShape'}


def getEmptyShape():
    return {'offsets': {}, 'weights': {}, 'shapeIndex': None}


def setBlendShapeData(node,
                      shapeData,
                      inputIndex=0,
                      shapes=None):
    """
    sets the shape data onto a blendShape node.
    :param str node: the blendShape to add the shapes to.
    :param dict shapeData: the shape data to apply to the node.
    :param list shapes: if given, only modify given target names
    """
    nodeType = mc.nodeType(node)
    inputShape = mc.deformer(node, g=True, q=True)
    shapeAliasLookup = getShapeAliasLookup(node)

    if not 'shapes' in shapeData:
        print("procedureName" + ':  shapeData does not have a "shapes" key.  Returning now...')
        return

    for shapeAlias in shapeData['shapes']:
        if shapes and shapeAlias not in shapes:
            continue

        # read the information stored for this shape
        targetData = shapeData['shapes'][shapeAlias]
        targetOffsets = targetData["offsets"]
        targetWeights = targetData["weights"]
        shapeIndex = shapeAliasLookup.get(shapeAlias, None)

        # if the shape doesn't already exist, create it at the end of the list
        newShape = False
        if shapeIndex is None:
            newShape = True
            weightIndices = mc.getAttr(node + ".weight", mi=True)
            if weightIndices is None:
                shapeIndex = 0
            else:
                shapeIndex = weightIndices[-1] + 1

            mc.addMultiInstance(node + '.weight[' + str(shapeIndex) + ']')
            if shapeAlias[0] != '[':
                mc.aliasAttr(shapeAlias.strip(), node + ".weight[" + str(shapeIndex) + "]")

        # iterate through the offset dictionary
        pointList = []
        componentList = []
        #speed optimization
        shapeComponentsToUse = {}
        
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
    nodeType = mc.nodeType(node)
    inputShape = mc.deformer(node, g=True, q=True)
    shapeAliasLookup = getShapeAliasLookup(node)

    if not 'shapes' in shapeData:
        print("procedureName" + ':  shapeData does not have a "shapes" key.  Returning now...')
        return

    for shapeAlias in shapeData['shapes']:
        if shapes and shapeAlias not in shapes:
            continue

        # read the information stored for this shape
        targetData = shapeData['shapes'][shapeAlias]
        targetOffsets = targetData["offsets"]
        targetWeights = targetData["weights"]
        shapeIndex = shapeAliasLookup.get(shapeAlias, None)

        # if the shape doesn't already exist, create it at the end of the list
        newShape = False
        if shapeIndex is None:
            newShape = True
            weightIndices = mc.getAttr(node + ".weight", mi=True)
            if weightIndices is None:
                shapeIndex = 0
            else:
                shapeIndex = weightIndices[-1] + 1

            mc.addMultiInstance(node + '.weight[' + str(shapeIndex) + ']')
            if shapeAlias[0] != '[':
                mc.aliasAttr(shapeAlias.strip(), node + ".weight[" + str(shapeIndex) + "]")

        # iterate through the offset dictionary
        pointList = []
        componentList = []
        #speed optimization
        shapeComponentsToUse = {}
        
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


def pose_rig(file_name):
    # Pose the character with given parameters
    f = open(input_path + file_name, 'r')
    reader = list(csv.reader(f))
    # set attribute values to model
    attribute = reader[0][1:]
    for line in reader[1:]:
        mover = line[0]
        # set each attribute
        for i, value in enumerate(line[1:]):
            # check the attribute is locked or not
            if mc.getAttr(mover + '.' + attribute[i], l=True):
                continue
            mc.setAttr(mover + '.' + attribute[i], float(value))
    f.close()


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


def retreive_data(curr_data, filename):
    # output data to csv files
    new_path = temp_path + filename[:-4]
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    for key, item in curr_data.items():
        f = open(new_path + '/' + key + ".csv", 'w')
        csv_writer = csv.writer(f)
        for k, it in item.items():
            csv_writer.writerow([k] + it)
        f.close()


def reconstruction(training_type='differential', ground_truth=False):
    # set the deformers
    BLENDSHAPE = TEMP_BS_NODE
    mesh = mc.deformer(BLENDSHAPE, query=True, geometry=True)[0]
    print(mesh)

    # read in data for reconstruction
    pose_rig = read_in_rig(input_path + filename)
    pose_data = read_in_data(temp_path + filename[:-4])

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

    setShapeData(BLENDSHAPE, shape_data, shapes=[TEMP_TARGET])
    mc.setAttr(BLENDSHAPE + '.' + TEMP_TARGET, 1.0)


# Use this when need batch generating
# filenames = []
# for root, dirs, files in os.walk(input_path):
#     filenames.append(files)

# read in the csv file
pose_rig(filename)

# TODO: model is posed randomly
# need to extract other information like mesh info
curr_data = {}
connectionMap = {}
anchorIndex = []
joint_dict = {}

curr_data['anchorPoints'] = {}
curr_data['differentialOffset'] = {}
curr_data['worldPos'] = {}
curr_data['worldOffset'] = {}
curr_data['localOffset'] = {}
curr_data['jointWorldMatrix'] = {}
curr_data['jointWorldQuaternion'] = {}
curr_data['jointLocalMatrix'] = {}
curr_data['jointLocalQuaternion'] = {}

# Read in the connection map
f = open(topology_path, 'r')
reader = csv.reader(f)
content = list(reader)
for line in content:
    if len(line) > 1:
        connectionMap[int(line[0])] = [int(c) for c in line[1:]]
    else:
        connectionMap[int(line[0])] = []
f.close()

# Read in the anchor points
f = open(anchor_path, 'r')
reader = csv.reader(f)
content = list(reader)
anchorIndex = [int(line[0]) for line in content[1:]]
f.close()

# Read in the joint relations
f = open(joint_path, 'r')
reader = csv.reader(f)
content = list(reader)
for line in content:
    joint_dict[line[1]] = line[2]
f.close()

# Get the joint worldMatrix and worldQuaternion
world_mats = {}
for jnt in joint_dict:
    world_mat = mc.getAttr(jnt + '.worldMatrix[0]')
    world_mats[jnt] = world_mat

    wm = om.MTransformationMatrix(om.MMatrix(world_mat))
    quaternion = wm.rotation(asQuaternion=True)

    temp = [
        world_mat[0], world_mat[1], world_mat[2],
        world_mat[4], world_mat[5], world_mat[6],
        world_mat[8], world_mat[9], world_mat[10],
        world_mat[12], world_mat[13], world_mat[14]
    ]
    curr_data['jointWorldMatrix'][jnt] = [round(i, PRECISION) for i in temp]

    temp = [
        quaternion[0], quaternion[1], quaternion[2], quaternion[3],
        world_mat[12], world_mat[13], world_mat[14]
    ]
    curr_data['jointWorldQuaternion'][jnt] = [round(i, PRECISION) for i in temp]

# Get the joint local matrix and local quaternion
for jnt in joint_dict:
    parent = joint_dict[jnt]
    local_mat = None
    if parent:
        parent_mat = om.MMatrix(world_mats[parent])
        world_mat = om.MMatrix(world_mats[jnt])
        local_mat = world_mat * parent_mat.inverse()
    else:
        local_mat = om.MMatrix(world_mats[jnt])

    lm = om.MTransformationMatrix(local_mat)
    quaternion = lm.rotation(asQuaternion=True)

    temp = [
        local_mat[0], local_mat[1], local_mat[2],
        local_mat[4], local_mat[5], local_mat[6],
        local_mat[8], local_mat[9], local_mat[10],
        local_mat[12], local_mat[13], local_mat[14]
    ]
    curr_data['jointLocalMatrix'][jnt] = [round(i, PRECISION) for i in temp]

    temp = [quaternion[0], quaternion[1], quaternion[2], quaternion[3],
            local_mat[12], local_mat[13], local_mat[14]]
    curr_data['jointLocalQuaternion'][jnt] = [round(i, PRECISION) for i in temp]


# Get the mesh vertex world information throuth following steps
meshShape, positions, curr_data['worldPos'] = get_worldPos(MESH, PRECISION)
# Create a duplicate
duplicate = mc.duplicate(
        MESH,
        name=MESH + 'Dup',
        upstreamNodes=False,
        returnRootsOnly=True
    )[0]
# Create deformers
deformers = prep_mesh(MESH)
deformer_env_dict = {}
# shutdown all deformers except for skin clusters
for deformer in deformers:
    dtype = mc.nodeType(deformer)
    if dtype not in SKIN_TYPES:
        deformer_env_dict[deformer] = mc.getAttr(deformer + '.envelope')
        mc.setAttr(deformer + '.envelope', 0.0)
# Get mesh linear postions
linear_pos, curr_data['worldOffset'] = get_worldOffset(MESH, PRECISION, meshShape, positions)

# Get local offset before linear skin blending
offsets = get_localOffset(MESH, duplicate, TEMP_BS_NODE, TEMP_TARGET)
vertex_count = len(curr_data['worldPos'])
for i in range(vertex_count):
    offset = offsets.get(i, [0.0, 0.0, 0.0])
    curr_data['localOffset'][i] = [round(data, PRECISION) for data in offset]

# Generate the differential offset data
for i in range(len(connectionMap)):
    neighbors = connectionMap[i]
    valence = float(len(neighbors))
    new_coord = offsets.get(i, [0.0, 0.0, 0.0])
    neighbor_values = [0.0, 0.0, 0.0]
    for neighbor in neighbors:
        nb_coord = offsets.get(neighbor, [0.0, 0.0, 0.0])
        neighbor_values[0] += nb_coord[0]
        neighbor_values[1] += nb_coord[1]
        neighbor_values[2] += nb_coord[2]

    x = new_coord[0] - neighbor_values[0] / valence
    y = new_coord[1] - neighbor_values[1] / valence
    z = new_coord[2] - neighbor_values[2] / valence

    offset = [x, y, z]
    curr_data['differentialOffset'][i] = [round(data, PRECISION) for data in offset]

# Get the anchor points data
for anchor in anchorIndex:
    offset = offsets.get(anchor, [0.0, 0.0, 0.0])
    curr_data['anchorPoints'][anchor] = [round(data, PRECISION) for data in offset]

# Restore the rig
for deformer in deformers:
    dtype = mc.nodeType(deformer)
    if dtype not in SKIN_TYPES:
        mc.setAttr(deformer + '.envelope', deformer_env_dict[deformer])
mc.delete(duplicate)

retreive_data(curr_data, filename)

# reconstruction(training_type="local_offset")
