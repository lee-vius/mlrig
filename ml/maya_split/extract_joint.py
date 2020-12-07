import maya.cmds as mc
import csv

#folder_path = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/"
folder_path = "D:\ACG\project\ml\maya_split\gen_data/"


def get_joint_relation(filename="joint_relation.csv"):
    joints = mc.ls(type="joint")
    f = open(folder_path + filename, 'w')
    csv_writer = csv.writer(f)
    for index, joint in enumerate(joints):
        parent = mc.listRelatives(joint, parent=True, type="joint")
        csv_writer.writerow([index] + [joint] + [parent])
    f.close()

