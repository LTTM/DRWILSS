import os
import numpy as np
from tqdm import tqdm
import json
import sys
import tasks


def get_classes(dataset):
    if dataset == "coco-voc":
        return [
            "unlabeled",
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "street sign",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "hat",
            "backpack",
            "umbrella",
            "shoe",
            "eye glasses",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "plate",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "mirror",
            "dining table",
            "window",
            "desk",
            "toilet",
            "door",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "blender",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
            "hair brush",
        ]
    elif dataset == "voc":
        return [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorbike",
            "person",
            "potted plant",
            "sheep",
            "sofa",
            "train",
            "tv monitor",
        ]
    else:
        raise ValueError(f"Dataset {dataset} not supported! Choose between 'coco-voc' and 'voc'.")


class wilson_opts_obj():
    def __init__(self, dataset, task, overlap, step):
        self.dataset = dataset
        self.task = task
        self.overlap = overlap
        self.task_and_ov = f"{task}-ov" if overlap else task
        self.step = step
        self.test = True
        self.norm_act = "iabn_sync"
        self.no_pretrained = False
        self.pooling = 32
        if self.dataset == "coco-voc":
            self.backbone = "wider_resnet38_a2"
            self.output_stride = 8
        elif self.dataset == "voc":
            self.backbone = "resnet101"
            self.output_stride = 16
        else:
            raise ValueError(f"Dataset {dataset} not supported! Choose between 'coco-voc' and 'voc'.")


def get_image_names_and_onehots(opts):
    idxs_path = f"data/{opts.dataset}/{opts.task_and_ov}/train-{opts.step}.npy"
    idxs = np.load(idxs_path)
    if opts.dataset == "coco-voc":
        with open(f"data/coco/split/train.txt") as f:
            lines = f.readlines()
            lines = [l.strip() for l in lines]
            image_names = np.array(lines)[idxs]
        onehot_labels = np.load(f"data/coco/1h_labels_train.npy")
    elif opts.dataset == "voc":
        with open(f"data/voc/splits/train_aug.txt") as f:
            lines = f.readlines()
            lines = [l.strip().split(" ")[0][1:] for l in lines]
            image_names = np.array(lines)[idxs]
        onehot_labels = np.load(f"data/voc/voc_1h_labels_train.npy")
    else:
        raise ValueError(f"Dataset {opts.dataset} not supported! Choose between 'coco-voc' and 'voc'.")
    onehot_labels = onehot_labels[idxs]
    good_indices = np.where(np.sum(onehot_labels, axis=1) != 0)[0]  # some images have all-0 labels because they only contain an "ignore_label" (e.g. "street sign" (12), "hair brush" (91))
    if len(good_indices) < len(image_names):
        print(f"Filtered out {len(image_names) - len(good_indices)} images with no valid labels.")
    return image_names[good_indices].astype(object), onehot_labels[good_indices]


def get_captions(image_names, onehots, step_classes, shuffle, prompt_skeleton):
    captions = []
    present_classes_list = []
    for i, image_name in enumerate(tqdm(image_names)):
        present_classes = np.array(step_classes)[onehots[i] == 1]
        num_present_classes = len(present_classes)
        if num_present_classes == 0:
            raise ValueError(f"No classes present in image {image_name}, aborting...")  # some images seem to be broken
        if num_present_classes == 1:
            if prompt_skeleton is not None:
                prompt = prompt_skeleton.replace("CLASS", present_classes[0])
            else:
                prompt = f"photo of a {present_classes[0]}"
        else:
            if shuffle:
                np.random.shuffle(present_classes)  # shuffle to avoid bias for first class
            cl_string = ""
            for pc in present_classes[:-1]:
                cl_string += pc + ", a "
            cl_string += "and a " + present_classes[-1]
            cl_string = cl_string.replace(", a and a", " and a")
            if prompt_skeleton is not None:
                prompt = prompt_skeleton.replace("CLASS", cl_string)
            else:
                prompt = f"photo of a {cl_string}"
        captions.append(prompt)
        present_classes_list.append(present_classes)
    return captions, present_classes_list


def copy_training_data(image_names, captions, present_classes_list, opts):
    # for i in range(10):
    #     print(image_names[i])
    #     print(captions[i])
    if opts.dataset == "coco-voc":
        real_img_path = f"data/coco/images/train2017"
        real_img_suffix = ".jpg"
    elif opts.dataset == "voc":
        real_img_path = f"data/voc"
        real_img_suffix = ""
    metadata = []
    for i, image_name in enumerate(tqdm(image_names)):
        caption = captions[i]
        present_classes = present_classes_list[i]
        os.system(f"cp {real_img_path}/{image_name}{real_img_suffix} lora_prep_data_seed_{seed}/{opts.task_and_ov}/{opts.step}/{image_name.split('/')[-1]}{real_img_suffix}")
        metadata.append({
            "file_name": f"{image_name.split('/')[-1]}{real_img_suffix}",
            "text": caption,
            "present_classes": present_classes.tolist()
            })
    with open(f"lora_prep_data_seed_{seed}/{opts.task_and_ov}/{opts.step}/metadata.jsonl", "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    dataset = sys.argv[1]  # "coco-voc" or "voc"
    task = sys.argv[2]  # e.g. "15-5"
    overlap = sys.argv[3] == "overlap"  # "disjoint" or "overlap"
    step = sys.argv[4]  # int or "all"

    shuffle = sys.argv[5] == "shuffle"  # "shuffle" or "noshuffle" of classes in captions
    mode = sys.argv[6]  # "debug" to test generation with a subset of the data, otherwise full generation is assumed

    prompt_skeleton = sys.argv[7]  # e.g. "photo of a CLASS"
    seed = sys.argv[8]

    np.random.seed(int(seed))
    
    # options, sanity checks and warnings
    if step == "all":
        step_list = list(tasks.tasks[dataset][task].keys())[:-1] # exclude last step
    else:
        step_list = [int(step)]
            
    for current_step in step_list:
        opts = wilson_opts_obj(
            dataset=dataset,
            task=task,
            overlap=overlap,
            step=current_step)
        opts.seed = seed
        
        os.makedirs(f"lora_prep_data_seed_{seed}/{opts.task_and_ov}/{opts.step}", exist_ok=True)

        classes = get_classes(dataset=dataset)
        step_class_inds = tasks.tasks[dataset][task][current_step].copy()
        if 0 in step_class_inds:
            step_class_inds.remove(0)  # no background
        if dataset == "coco-voc":
            step_classes = classes[1:]
        elif dataset == "voc":
            step_classes = [classes[i-1] for i in step_class_inds]

        print("Loading image names...")
        image_names, onehot_labels = get_image_names_and_onehots(opts)
        if dataset == "voc":
            onehot_labels = onehot_labels[:, np.array(step_class_inds) - 1]  # only keep classes for this step

        if mode == "debug":
            if dataset == "coco-voc":
                image_names = image_names[:4950]
                onehot_labels = onehot_labels[:4950]
            elif dataset == "voc":
                image_names = image_names[:256]
                onehot_labels = onehot_labels[:256]

        print("Preparing captions...")
        captions, present_classes_list = get_captions(
            image_names,
            onehot_labels,
            step_classes,
            shuffle,
            prompt_skeleton=prompt_skeleton
        )

        print("Copying training data...")
        copy_training_data(image_names, captions, present_classes_list, opts)
        
        print("Done with step", current_step)
    print("Done with all steps!")
