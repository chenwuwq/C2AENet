import os
import re

input_path = "denoise_dataset/mesh_denoising_data/Kinect_v2/test/original/"
out_path = "denoise_dataset/mesh_denoising_data/Kinect_v2/test/original_mesh/"
objs = os.listdir(input_path)

for i in objs:
    file = open(input_path + i, "r")
    lines = file.readlines()
    vertices = []
    faces = []
    for line in lines:
        if line.startswith("v "):
            vertices.append(list(map(float, line.split()[1:])))
        elif line.startswith("f "):
            f = list(map(lambda x: int(x.split('/')[0]), line.split()[1:]))
            f = [f[0]-1, f[1]-1, f[2]-1]
            faces.append(f)

    with open(out_path + i.split(".obj")[0] + ".off", "w") as out_file:
        out_file.write("OFF\n")
        out_file.write('{} {} {}\n'.format(len(vertices), len(faces), 0))
        for vertex in vertices:
            out_file.write(' {} {} {}\n'.format(*vertex))
        for face in faces:
            out_file.write('3 {} {} {}\n'.format(*face))