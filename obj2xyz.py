import os
import re

input_path = "denoise_dataset/mesh_denoising_data/Kinect_v2/test/original/"
out_path = "denoise_dataset/mesh_denoising_data/Kinect_v2/test/original_xyz/"
objs = os.listdir(input_path)

for i in objs:
    file = open(input_path + i, "r")
    vertices = []
    for line in file:
        if line.startswith("v "):
            match = re.match(r'v (\S+) (\S+) (\S+)', line)
            if match:
                vertices.append([float(match.group(1)), float(match.group(2)), float(match.group(3))])
    with open(out_path + i.split(".obj")[0] + ".xyz", "w") as out_file:
        for vertex in vertices:
            out_file.write('{} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))
