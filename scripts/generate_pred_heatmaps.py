import numpy as np
import torch
import os
import sys
import time
from pathlib import Path
from heatmap_estimator import HeatmapEstimator

projections_dir = Path(sys.argv[1])
models_dir = sys.argv[2]
heatmap_dir = projections_dir.parent.absolute() / "heatmap_preds"

planes = ["XY", "YZ", "ZX"]

device = 'cuda'
models = []
for plane in planes:
    model = HeatmapEstimator()
    model.load_state_dict(torch.load(models_dir + "/" + plane + ".pth"))
    model.to(device)
    model.eval()
    models.append(model)

total_time = 0
num_evals = 0

def gen_heatmap(file, data_dir, output_dir, plane):
    path = os.path.join(data_dir, file)
    depth = torch.zeros(96, 96)
    with open(path) as f:
        lines = f.readlines()
        for y in range(96):
            line = lines[y].split(' ')
            for x in range(96):
                depth[y, x] = float(line[x])
    depth = torch.unsqueeze(torch.unsqueeze(depth.to(device), axis=0), axis=0)

    heatmap_path = os.path.join(output_dir, file).replace("projection.txt", "temp")
    with torch.no_grad():
        # Time evaluation
        # global total_time, num_evals

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()
        pred = models[plane](depth)
        # end.record()

        # torch.cuda.synchronize()
        # total_time += start.elapsed_time(end)
        # num_evals += 1

        pred = torch.squeeze(pred).cpu()
        for j in range(21):
            heatmap_pred = pred[j]
            joint = str(j).zfill(2)
            joint_path = heatmap_path.replace("temp", "heatmap-" + joint + ".txt")
            with open(joint_path, 'w') as f:
                for y in range(heatmap_pred.shape[0]):
                    line = ""
                    for x in range(heatmap_pred.shape[1]):
                        line += str(heatmap_pred[y, x].item()) + " "
                    line += "\n"
                    f.write(line)


for subject in sorted(os.listdir(projections_dir)):
    subject_dir = projections_dir / subject
    for gesture in sorted(os.listdir(subject_dir)):
        print("Computing gesture", gesture)
        data_dir = subject_dir / gesture
        output_dir = heatmap_dir / subject / gesture

        if os.path.exists(output_dir):
            print(f'{output_dir} already exists')
        else:
            os.makedirs(output_dir)

        for file in sorted(os.listdir(data_dir)):
            if "-projection.txt" in file:
                for i in range(len(planes)):
                    if planes[i] in file:
                        gen_heatmap(file, data_dir, output_dir, i)
        # Only time for gesture 1
        # break
    # Only run on subject P0
    break

# Print average eval time
# print("Average evaluation time", total_time / num_evals)