import os
import csv
import numpy
import pandas


def read_in_data(filepath):
    # get the data files to read
    filenames = []
    for root, dirs, files in os.walk(filepath):
        filenames.append(files)

    # read in the result
    pose_data = {}
    for filename in filenames:
        # for each file, read in as a dict
        f = open(filepath + filename, 'r')
        temp = {}
        reader = csv.reader(f)
        content = list(reader)
        if filename[:-4] in ['anchorPoints', 'differentialOffset', 'localOffset', 'worldOffset', 'worldPos']:
            for line in content:
                temp[int(line[0])] = [float(coord) for coord in line[1:]]
        elif filename[:-4] in ['jointLocalMatrix', 'jointLocalQuaternion', 'jointWorldMatrix', 'jointWorldQuaternion']:
            for line in content:
                temp[line[0]] = [float(coord) for coord in line[1:]]
        else:
            for line in content:
                temp[line[0]] = line[1:]
        f.close()
        pose_data[filename[:-4]] = temp

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


if __name__ == '__main__':
    filepath = "D:\ACG\project\ml\maya_split\gen_data\mover_rigged/rigged0.csv"
    pose_rig = read_in_rig(filepath)
    print(pose_rig)
