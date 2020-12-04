import maya.cmds as mc
import maya.api.OpenMaya as om
import csv
import re

folder_path = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/"

def get_anchor_points(filename="face_anchor.csv", mesh="Mery_geo_cn_body.vtx"):
    # Need manually choose anchor points
    anchors = mc.ls(sl=True)[1:]
    title = mesh
    f = open(folder_path + filename, 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow([title])
    for anchor in anchors:
        index = re.findall(r"\d+", anchor)[0]
        csv_writer.writerow([index])
    f.close()
        