# This script prepares data splits for other tasks (10-1, 15-1, 10-5).
# Overlap saves all images that have at least one of the current classes present from the global train split.
# Disjoint only saves images that have no future classes present.
# The script assumes that tasks.py contains the correct class splits.

import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import tasks


if __name__ == "__main__":
    voc_root = "data/voc"
    task = input("Enter task (e.g. 10-10): ")
    overlap = input("Is it overlap? (y/n): ")
    if overlap == "y":
        task_and_ov = f"{task}-ov"
    else:
        task_and_ov = f"{task}"
    task_dir = f"data/voc/{task_and_ov}"
    os.makedirs(task_dir, exist_ok=True)
    task_dict = tasks.tasks["voc"][task]

    with open("data/voc/splits/train_aug.txt", "r") as f:
        file_names = [x[:-1].split(' ') for x in f.readlines()]
    print(len(file_names))
    # images = [(os.path.join(voc_root, x[0][1:]), os.path.join(voc_root, x[1][1:])) for x in file_names]
    images = [os.path.join(voc_root, x[1][1:]) for x in file_names]
    # print(images)
    uniques = []
    for image in tqdm(images):
        img = np.array(Image.open(image))
        # print(img.shape)
        unqs = np.unique(img)
        unqs = unqs[unqs > 0]
        unqs = unqs[unqs < 255]
        uniques.append(unqs)

    if "ov" in task_and_ov:  # overlap
        for step, classes in task_dict.items():
            idxs = []
            for i, unqs in enumerate(uniques):
                for c in classes:
                    if c in unqs:
                        idxs.append(i)
                        break
            step_idxs = np.array(idxs, dtype=np.int64)
            np.save(f"{task_dir}/train-{step}.npy", step_idxs)
            print(f"Saved {len(step_idxs)} images for step {step}.")
    else:  # disjoint
        for step, classes in task_dict.items():
            max_class = max(classes)
            idxs = []
            for i, unqs in enumerate(uniques):
                max_unq = max(unqs)
                if max_unq <= max_class:  # make sure no future classes present
                    for c in classes:
                        if c in unqs:
                            idxs.append(i)
                            break
            step_idxs = np.array(idxs, dtype=np.int64)
            np.save(f"{task_dir}/train-{step}.npy", step_idxs)
            print(f"Saved {len(step_idxs)} images for step {step}.")
