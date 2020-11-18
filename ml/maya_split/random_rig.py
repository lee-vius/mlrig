import csv
import scipy.stats as stats

rest_rig_path = "/Users/levius/Desktop/高级图像图形学/项目/code/ml/maya_split/gen_data/mover_range.csv"

def read_rest_rig():
    # read in the csv file
    f = open(rest_rig_path, 'r')
    reader = csv.reader(f)
    result = list(reader)
    # construct a dictionary
    rig_value = {}
    header = result[0]
    for line in result[1:]:
        rig_value[line[0]] = []
        data = line[1:]
        for i in range(len(data) / 2):
            rig_value[line[0]].append([data[2*i], data[2*i + 1]])
    
    return header, rig_value


def gen_random_rig(distri_scale=1.0):
    # get the rig value
    header, rest_value = read_rest_rig()
    distribution = stats.truncnorm(-1*distri_scale, distri_scale, loc=0.0, scale=1.0)


if __name__ == "__main__":
    gen_random_rig()