import os
import torch
import json
from tqdm import tqdm
from PIL import Image
import clip
import numpy as np
import sys
import tasks


def select_top_k(task_ov, step, model, preprocess, top_k, seed):
    if os.path.exists(f"lora_prep_data_seed_{seed}/{task_ov}_top_{top_k}/{step}"):
        print("Found existing top-k selection, deleting...")
        os.system(f"rm -rf lora_prep_data_seed_{seed}/{task_ov}_top_{top_k}/{step}")
    classes = []
    class_to_imgs = {}
    with open(f"lora_prep_data_seed_{seed}/{task_ov}/{step}/metadata.jsonl", "r") as f:
        for line in f.read().splitlines():
            item = json.loads(line)
            for cl in item["present_classes"]:
                if cl not in classes:
                    classes.append(cl)
                    class_to_imgs[cl] = []
                class_to_imgs[cl].append(item["file_name"])
    classes = sorted(classes)

    prototype_dict = {}
    with torch.no_grad():
        for cl in tqdm(classes, leave=True, desc=f"Step {step}"):
            n_imgs = len(class_to_imgs[cl])
            embeds = torch.zeros(size=(n_imgs, 512))
            for i, img_name in enumerate(tqdm(class_to_imgs[cl], leave=False)):
                img = preprocess(Image.open(f"lora_prep_data_seed_{seed}/{task_ov}/{step}/{img_name}").convert("RGB")).unsqueeze(0).to("cuda")
                embed = model.encode_image(img)
                embeds[i] = embed
            prototype = embeds.mean(dim=0)
            cossims = torch.nn.functional.cosine_similarity(embeds, prototype)
            inds = torch.argsort(cossims, descending=True, stable=True)
            top_k_cossims = cossims[inds][:top_k]
            top_k_img_names = np.array(class_to_imgs[cl])[inds][:top_k]
            cl_underscored = cl.replace(" ", "_")
            os.makedirs(f"lora_prep_data_seed_{seed}/{task_ov}_top_{top_k}/{step}/train/{cl_underscored}")
            for top_img in top_k_img_names:
                os.system(f"cp lora_prep_data_seed_{seed}/{task_ov}/{step}/{top_img} lora_prep_data_seed_{seed}/{task_ov}_top_{top_k}/{step}/train/{cl_underscored}/{top_img}")
            with open(f"lora_prep_data_seed_{seed}/{task_ov}/{step}/metadata.jsonl", "r") as f:
                lines = f.read().splitlines()
            with open(f"lora_prep_data_seed_{seed}/{task_ov}_top_{top_k}/{step}/train/metadata.jsonl", "a") as f:
                for line in lines:
                    item = json.loads(line)
                    if item["file_name"] in top_k_img_names:
                        item["file_name"] = f"{cl_underscored}/{item['file_name']}" 
                        f.write(json.dumps(item) + "\n")
            prototype_dict[cl_underscored] = prototype.cpu().tolist()
    with open(f"lora_prep_data_seed_{seed}/{task_ov}_top_{top_k}/{step}/prototypes.json", "w") as f:
        json.dump(prototype_dict, f)


if __name__ == "__main__":
    dataset = sys.argv[1]  # "coco-voc" or "voc"
    task = sys.argv[2]  # e.g. "15-5"
    overlap = sys.argv[3] == "overlap"  # "disjoint" or "overlap"
    step = sys.argv[4]  # int or "all"
    top_k = int(sys.argv[5])  # e.g. 10, how many images to select per class
    seed = int(sys.argv[6])

    model, preprocess = clip.load("ViT-B/32", device="cuda")  # latent size 512

    task_ov = f"{task}-ov" if overlap else task
    if step == "all":
        step_list = list(tasks.tasks[dataset][task].keys())[:-1] # exclude last step
    else:
        step_list = [int(step)]

    print(f"Preparing {dataset}-{task_ov}")
    for current_step in step_list:
        select_top_k(task_ov, current_step, model, preprocess, top_k, seed)

    print("Done!")
