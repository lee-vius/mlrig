import numpy as np
import csv
import os


def read_in_data(filepath):
    # get the data files to read
    files = []
    for _, _, fs in os.walk(filepath):
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


def cal_square_loss(mat1, mat2):
    result = np.square(mat1 - mat2)
    result = result.sum(axis=1)
    return np.sqrt(result)


def convert_array(dict_x):
    result = []
    for i, key in enumerate(dict_x):
        result.append(dict_x[key])
    return np.array(result)



if __name__ == "__main__":
    filepath1 = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/test_data/rigged0_accu"
    filepath2 = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/test_data/rigged0_local"
    filepath3 = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/test_data/rigged0_diff"
    pose_data_accu = read_in_data(filepath1)
    pose_data_local = read_in_data(filepath2)
    pose_data_diff = read_in_data(filepath3)
    
    pos_accu = convert_array(pose_data_accu['worldPos'])
    pos_local = convert_array(pose_data_local['worldPos'])
    pos_diff = convert_array(pose_data_diff['worldPos'])

    loss1 = cal_square_loss(pos_accu, pos_local)
    loss2 = cal_square_loss(pos_accu, pos_diff)
    loss3 = cal_square_loss(pos_local, pos_diff)

    print("accu - local: " + str(np.mean(loss1)))
    print("accu - diff: " + str(np.mean(loss2)))
    print("diff - local: " + str(np.mean(loss3)))

    print("accu - local - max: " + str(np.max(loss1)))
    print("accu - diff - max: " + str(np.max(loss2)))
    print("diff - local - max: " + str(np.max(loss3)))
