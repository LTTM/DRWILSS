from PIL import Image
import numpy as np
import os
import shutil
from tqdm import tqdm
from dataset.voc import VOCSegmentation


for split in ["train2017", "val2017"]:
    raw_dir = f"data/coco/annotations_raw/{split}"
    prep_dir = f"data/coco/annotations/{split}"

    # clear folder if exists
    if os.path.exists(prep_dir):
        print("Removing existing")
        shutil.rmtree(prep_dir)
    os.makedirs(prep_dir)

    # remap labels in the novel interval (+1 for Stuff, +1 and stuff on 0 for objects)
    prep_mapping = np.zeros((256,), dtype=np.int8)
    for i, cl in enumerate(range(91)):
        prep_mapping[cl] = i + 1
    prep_mapping[255] = -1
    prep_transform = lambda x: Image.fromarray(prep_mapping[x])

    for f in tqdm(os.listdir(raw_dir)):
        lbl = Image.open(os.path.join(raw_dir, f))
        lbl = prep_transform(lbl)
        lbl.save(os.path.join(prep_dir, f))


coco_map = {
    0: 0,
    1: 5,
    2: 2,
    3: 16,
    4: 9,
    5: 44,
    6: 6,
    7: 3,
    8: 17,
    9: 62,
    10: 21,
    11: 67,
    12: 18,
    13: 19,
    14: 4,
    15: 1,
    16: 64,
    17: 20,
    18: 63,
    19: 7,
    20: 72,
    255: 255
}

class LabelTransform:
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, x):
        return Image.fromarray(self.mapping[x])

cocovoc_mapping = np.zeros((256,), dtype=np.uint8)
for k,v in coco_map.items():
    cocovoc_mapping[k] = v
remap = LabelTransform(cocovoc_mapping)

train_data = VOCSegmentation("data", train=True)
os.makedirs(f"data/voc/SegmentationClassAugAsCoco", exist_ok=True)

split_txt = []
for ip, lp in tqdm(train_data.images):
    lbl = Image.open(lp)
    idx = lp.split("/")[-1][:-4]
    new_path = f"data/voc/SegmentationClassAugAsCoco/{idx}.png"
    new_lbl = remap(lbl)
    new_lbl.save(new_path, "PNG")

# repeat for validation images
val_data = VOCSegmentation("data", train=False)

split_txt = []
for ip, lp in tqdm(val_data.images):
    lbl = Image.open(lp)
    idx = lp.split("/")[-1][:-4]
    new_path = f"data/voc/SegmentationClassAugAsCoco/{idx}.png"
    new_lbl = remap(lbl)
    new_lbl.save(new_path, "PNG")
