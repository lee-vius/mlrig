import csv
import scipy.stats as stats

# input file -- store the range of each attribute
rest_rig_path = "D:\ACG\project\ml\maya_split\gen_data\mover_range_fix.csv"
# output path -- a folder to store rigged results
out_path = "D:\ACG\project\ml\maya_split\gen_data\mover_rigged/"

def read_rest_rig(input_file=rest_rig_path):
    # read in the csv file
    f = open(input_file, 'r')
    reader = csv.reader(f)
    result = list(reader)
    # construct a dictionary
    rig_value = {}
    header = result[0]
    for line in result[1:]:
        rig_value[line[0]] = []
        data = line[1:]
        for i in range(int(len(data) / 2)):
            rig_value[line[0]].append([float(data[2*i]), float(data[2*i + 1])])
    
    return header, rig_value


def gen_random_rig(distri_scale=2.0, input_file=rest_rig_path, file_path=out_path, file_code=0):
    # get the rig value
    _, rig_value = read_rest_rig(input_file)
    distribution = stats.truncnorm(-1*distri_scale, distri_scale, loc=0.0, scale=0.5)

    attributes = ['translateX', 'translateY', 'translateZ',
              'rotateX', 'rotateY', 'rotateZ',
              'scaleX', 'scaleY', 'scaleZ']

    
    # generate random value of each attr of each mover
    data = {}
    for _, mover in enumerate(rig_value):
        rig_range = rig_value[mover]
        distr = distribution.rvs(len(rig_range))
        new_value = []

        # iterate over each attr
        for i, attr in enumerate(rig_range):
            # attr[0] is minimal and attr[1] is maximal
            value = attr[0] + (attr[1] - attr[0]) * (distr[i] + distri_scale) / (distri_scale * 2.0)
            new_value.append(value)
        
        data[mover] = new_value
    
    # output csv file
    f = open(file_path + 'rigged' + str(file_code) + '.csv', 'w')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['mover'] + attributes)
    for i, key in enumerate(data):
        csv_writer.writerow([key] + data[key])
    f.close()

if __name__ == "__main__":
    print("running")
    for i in range(100):
        gen_random_rig(file_code=i)