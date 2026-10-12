import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from pytorch_lightning import seed_everything
import contextlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import pickle
import sys
import copy
from diffusers.utils.torch_utils import is_compiled_module
from diffusers import UNet2DConditionModel, DiffusionPipeline, AutoencoderKL
from torch.nn import DataParallel
from segmentation_module import make_model
from accelerate import PartialState
import tasks
from dataset import transform
from torchvision.transforms import Resize


_ = torch.set_grad_enabled(False)


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


_transform = transform.Compose([
    # transform.Resize(size=512),  # Not necessary, gen_imgs are 512x512 already
    # transform.CenterCrop(size=512),
    transform.ToTensor(),
    transform.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])


class wilson_opts_obj():
    def __init__(self, dataset, task, overlap, step, pseudolabeler_name):
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
        self.ckpt = f"./checkpoints/step/{self.dataset}-{self.task_and_ov}/{pseudolabeler_name}_{step}.pth"


def load_pseudolabel_model(opts):
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        net_dict = model.state_dict()
        pretrained_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state"].items() if
            (k.replace("module.", "") in net_dict) and
            (v.shape == net_dict[k.replace("module.", "")].shape)}
        assert pretrained_dict.keys() == net_dict.keys(), "STATE DICTS NOT COMPATIBLE!"
        net_dict.update(pretrained_dict)
        model.load_state_dict(net_dict)
        del checkpoint
        for par in model.parameters():
            par.requires_grad = False
        model.eval()
    return model


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
        if distributed_state.is_main_process:
            print(f"Filtered out {len(image_names) - len(good_indices)} images with no valid labels.")
    return image_names[good_indices].astype(object), onehot_labels[good_indices]


def get_captions(replay_root, image_names, onehots, step_classes, shuffle, prompt_skeleton, opts):
    metadata = {}
    captions = []
    class_counts = {c: 0 for c in step_classes}
    if distributed_state.is_main_process:
        pbar = enumerate(tqdm(image_names))
    else:
        pbar = enumerate(image_names)
    for i, image_name in pbar:
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

        captions += [prompt]
        metadata[image_name] = {"caption": prompt}
        metadata[image_name]["classes"] = present_classes
        metadata[image_name]["real_name"] = image_name
        metadata[image_name]["gen_name"] = f"{i:06}.jpg"
        for pc in present_classes:
            class_counts[pc] += 1
    # saving class counts file
    with open(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/class_counts.pkl", "wb") as f:
        pickle.dump(class_counts, f)
    return captions, metadata


def unwrap_model(model):
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def get_pipeline_and_args(gen_model, opts):
    img_gen_args = {}
    if gen_model == "sdxl":
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipeline.unet.requires_grad_(False)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
    elif gen_model == "sd21":
        unet = UNet2DConditionModel.from_pretrained(
            "sd2-community/stable-diffusion-2-1-base",
            subfolder="unet"
        )
        unet.requires_grad_(False)
        pipeline = DiffusionPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-1-base",
            unet=unwrap_model(unet),
            # text_encoder=unwrap_model(text_encoder),
            # tokenizer=unwrap_model(tokenizer),
            torch_dtype=torch.float32,
        )
        pipeline.unet.requires_grad_(False)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
    elif gen_model == "sd21_lora":
        unet = UNet2DConditionModel.from_pretrained(
            "sd2-community/stable-diffusion-2-1-base",
            subfolder="unet"
        )
        unet.requires_grad_(False)
        pipeline = DiffusionPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-1-base",
            torch_dtype=torch.float32,
        )
        pipeline.load_lora_weights(f"./lora_outputs_seed_{opts.seed}/{opts.task_and_ov}_top_{opts.top_k}/{opts.step}")
        pipeline.unet.requires_grad_(False)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
    elif gen_model == "sdxl_downscaled":
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae = vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipeline.unet.requires_grad_(False)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
    elif gen_model == "sdxl_lora":
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16
        )
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae = vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        pipeline.load_lora_weights(f"./lora_outputs_sdxl_seed_{opts.seed}/{opts.task_and_ov}_top_{opts.top_k}/{opts.step}")
        pipeline.unet.requires_grad_(False)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
    elif gen_model == "sdxl_turbo":
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/sdxl-turbo",
            variant="fp16",
            torch_dtype=torch.float16,
        )
        pipeline.unet.requires_grad_(False)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
        img_gen_args["guidance_scale"] = 0.0
    elif gen_model == "sd21_turbo":
        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            variant="fp16",
            torch_dtype=torch.float16,
        )
        pipeline.unet.requires_grad_(False)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        img_gen_args["guidance_scale"] = 0.0
    else:
        raise ValueError(f"Gen model {gen_model} not supported! Choose between 'sdxl' and 'sd21'.")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, img_gen_args


def generate_images(replay_root, metadata, img_gen_batch_size, seed, n_steps, res, neg_prompt, gen_model, opts):
    image_names = list(metadata.keys())
    pipeline, img_gen_args = get_pipeline_and_args(gen_model, opts)
    with torch.inference_mode():
        generator = torch.Generator()
        if seed is None:
            seed = 0
        generator = generator.manual_seed(seed)

        pipeline.to(distributed_state.device)

        n_devices = torch.cuda.device_count()
        if n_devices != distributed_state.num_processes:
            print(f"WARNING: Got {n_devices} GPUs, but started {distributed_state.num_processes} processes.")
        else:
            if distributed_state.is_main_process:
                print(f"Using {n_devices} GPUs for image generation.")

        big_batch_size = img_gen_batch_size * distributed_state.num_processes
        big_batch_list = [image_names[i : i + big_batch_size] for i in range(0, len(image_names), big_batch_size)]
        if distributed_state.is_main_process:
            pbar = enumerate(tqdm(big_batch_list, desc="batches"))
        else:
            pbar = enumerate(big_batch_list)
        for i, big_batch in pbar:
            with distributed_state.split_between_processes(big_batch) as img_names:
                prompts = []
                gen_names = []
                for im in img_names:
                    prompts.append(metadata[im]["caption"])
                    gen_names.append(metadata[im]["gen_name"])
                x_samples = pipeline(
                    prompt=prompts,
                    negative_prompt=neg_prompt,
                    num_inference_steps=n_steps,
                    height=res,
                    width=res,
                    generator=generator.manual_seed(seed + i + distributed_state.process_index * 100000),
                    **img_gen_args
                ).images
                if gen_model == "sdxl_downscaled" or gen_model == "sdxl_lora":
                    x_samples = [Resize((512, 512))(x_sample) for x_sample in x_samples]
                for j, x_sample in enumerate(x_samples):
                    x_sample.save(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/images/{gen_names[j]}", "JPEG", quality=95, optimize=True)

        distributed_state.wait_for_everyone()
        del pipeline
        del generator
        torch.cuda.empty_cache()


def generate_labels(replay_root, opts, step_class_inds, metadata, pseudolabel_batch_size, res, classes, gen_model, last_iter=False):
    image_names = list(metadata.keys())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if distributed_state.is_main_process:
        print(f"Loading checkpoint from {opts.ckpt}")
    pseudo_labeler = DataParallel(load_pseudolabel_model(opts))
    pseudo_labeler.to(device)

    big_batch_size = pseudolabel_batch_size * torch.cuda.device_count()
    big_batch_list = [image_names[i : i + big_batch_size] for i in range(0, len(image_names), big_batch_size)]
    if distributed_state.is_main_process:
        print(f"Using {torch.cuda.device_count()} GPUs for pseudolabel generation.")

    pseudolabels_1h = dict()
    unidentified = 0
    if distributed_state.is_main_process:
        pbar = tqdm(big_batch_list, leave=True)
    else:
        pbar = big_batch_list
    with torch.inference_mode():
        for big_batch in pbar:
            if gen_model == "sdxl_downscaled" or gen_model == "sdxl_lora":
                gen_imgs = torch.zeros((len(big_batch), 3, 512, 512))
            else:
                gen_imgs = torch.zeros((len(big_batch), 3, res, res))
            gen_names = []
            prompted_classes = []
            for i, im in enumerate(big_batch):
                gen_imgs[i] = _transform(Image.open(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/images/{metadata[im]['gen_name']}").convert("RGB")).to(device, dtype=torch.float32).unsqueeze(0)
                gen_names.append(metadata[im]["gen_name"])
                prompted_classes.append(metadata[im]["classes"])

            outputs = pseudo_labeler(gen_imgs)[0].cpu().numpy()  # outputs have shape (B, n_classes, 512, 512)
            pseudolabels = np.argmax(outputs, axis=1).astype(np.uint8)  # unique of this contains 0 for background
            if opts.dataset == "coco-voc":  # coco classes have a different order that does not align with incremental steps
                n_classes = 91
                skip_inds = [0]
                class_shift = 1
                class_map = step_class_inds
            elif opts.dataset == "voc":
                n_classes = 21
                skip_inds = [i for i in range(21) if i not in step_class_inds]
                class_shift = 0
                class_map = list(range(21))
            else:
                raise ValueError(f"Dataset {opts.dataset} not supported! Choose between 'coco-voc' and 'voc'.")
            
            for i, pseudolabel in enumerate(pseudolabels):
                pseudolabel_1h = np.zeros((n_classes), dtype=np.uint8)
                for present_class in np.unique(pseudolabel):
                    if present_class not in skip_inds:
                        pseudolabel_1h[class_map[present_class - class_shift] - class_shift] = 1
                # collect classes that were in the image prompt and found by the pseudolabeler
                all_found_classes_str = np.array([classes[i+class_shift] for i in np.arange(n_classes-(1-class_shift))[pseudolabel_1h[(1-class_shift):] == 1]])
                identified_all_flag = True
                for cl in prompted_classes[i]:  # iterate over classes that should be present
                    if cl not in all_found_classes_str:
                        identified_all_flag = False
                if identified_all_flag:
                    pseudolabels_1h[gen_names[i].replace(".jpg", ".png")] = pseudolabel_1h
                    pseudolabel = Image.fromarray(pseudolabel)
                    pseudolabel.save(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels/{gen_names[i].replace('.jpg', '.png')}")
                elif last_iter:
                    pseudolabels_1h[gen_names[i].replace(".jpg", ".png")] = pseudolabel_1h
                    pseudolabel = Image.fromarray(pseudolabel)
                    pseudolabel.save(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels/{gen_names[i].replace('.jpg', '.png')}")
                    unidentified += 1
                else:
                    #     print(f"WARNING: Not all classes identified for {img_name}, saving anyway")
                    unidentified += 1
        if distributed_state.is_main_process:
            with open(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/n_unidentified.txt", "a") as f:
                f.write(f"Number of identified images in this iteration: {unidentified} out of {len(image_names)}\n")
            print(f"Unidentified images: {unidentified} out of {len(image_names)}")
            with open(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels_1h.pkl", "wb") as f:
                pickle.dump(pseudolabels_1h, f)
        del pseudo_labeler
        torch.cuda.empty_cache()
    return unidentified


def regenerate_images(replay_root, metadata, img_gen_batch_size, seed, n_steps, res, neg_prompt, gen_model, opts, redo_iter):
    image_names = []
    already_good = sorted(os.listdir(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels"))
    for img_name in list(metadata.keys()):
        if metadata[img_name]["gen_name"].replace(".jpg", ".png") not in already_good:
            image_names.append(img_name)
    pipeline, img_gen_args = get_pipeline_and_args(gen_model, opts)
    with torch.inference_mode():
        generator = torch.Generator()
        if seed is None:
            seed = 0
        generator = generator.manual_seed(seed)

        pipeline.to(distributed_state.device)

        n_devices = torch.cuda.device_count()
        if n_devices != distributed_state.num_processes:
            print(f"WARNING: Got {n_devices} GPUs, but started {distributed_state.num_processes} processes.")
        else:
            if distributed_state.is_main_process:
                print(f"Using {n_devices} GPUs for image generation.")

        big_batch_size = img_gen_batch_size * distributed_state.num_processes
        big_batch_list = [image_names[i : i + big_batch_size] for i in range(0, len(image_names), big_batch_size)]
        if distributed_state.is_main_process:
            pbar = enumerate(tqdm(big_batch_list, desc="batches"))
        else:
            pbar = enumerate(big_batch_list)
        for i, big_batch in pbar:
            with distributed_state.split_between_processes(big_batch) as img_names:
                prompts = []
                gen_names = []
                for im in img_names:
                    prompts.append(metadata[im]["caption"])
                    gen_names.append(metadata[im]["gen_name"])
                x_samples = pipeline(
                    prompt=prompts,
                    negative_prompt=neg_prompt,
                    num_inference_steps=n_steps,
                    height=res,
                    width=res,
                    generator=generator.manual_seed(seed + i + 100000*distributed_state.process_index + 10000000*(redo_iter+1)),
                    **img_gen_args,
                ).images
                if gen_model == "sdxl_downscaled" or gen_model == "sdxl_lora":
                    x_samples = [Resize((512, 512))(x_sample) for x_sample in x_samples]
                for j, x_sample in enumerate(x_samples):
                    x_sample.save(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/images/{gen_names[j]}", "JPEG", quality=95, optimize=True)

        distributed_state.wait_for_everyone()
        del pipeline
        del generator
        torch.cuda.empty_cache()


def regenerate_labels(replay_root, opts, step_class_inds, metadata, pseudolabel_batch_size, res, classes, gen_model, last_iter=False):
    image_names = []
    already_good = sorted(os.listdir(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels"))
    for img_name in list(metadata.keys()):
        if metadata[img_name]["gen_name"].replace(".jpg", ".png") not in already_good:
            image_names.append(img_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if distributed_state.is_main_process:
        print(f"Loading checkpoint from {opts.ckpt}")
    pseudo_labeler = DataParallel(load_pseudolabel_model(opts))
    pseudo_labeler.to(device)
    
    big_batch_size = pseudolabel_batch_size * torch.cuda.device_count()
    big_batch_list = [image_names[i : i + big_batch_size] for i in range(0, len(image_names), big_batch_size)]
    if distributed_state.is_main_process:
        print(f"Using {torch.cuda.device_count()} GPUs for pseudolabel generation.")

    with open(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels_1h.pkl", "rb") as f:
        pseudolabels_1h = pickle.load(f)
    unidentified = 0
    if distributed_state.is_main_process:
        pbar = tqdm(big_batch_list, leave=True)
    else:
        pbar = big_batch_list
    with torch.inference_mode():
        for big_batch in pbar:
            if gen_model == "sdxl_downscaled" or gen_model == "sdxl_lora":
                gen_imgs = torch.zeros((len(big_batch), 3, 512, 512))
            else:
                gen_imgs = torch.zeros((len(big_batch), 3, res, res))
            gen_names = []
            prompted_classes = []
            for i, im in enumerate(big_batch):
                gen_imgs[i] = _transform(Image.open(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/images/{metadata[im]['gen_name']}").convert("RGB")).to(device, dtype=torch.float32).unsqueeze(0)
                gen_names.append(metadata[im]["gen_name"])
                prompted_classes.append(metadata[im]["classes"])

            outputs = pseudo_labeler(gen_imgs)[0].cpu().numpy()  # outputs have shape (B, n_classes, 512, 512)
            pseudolabels = np.argmax(outputs, axis=1).astype(np.uint8)  # unique of this contains 0 for background
            if opts.dataset == "coco-voc":
                n_classes = 91
                skip_inds = [0]
                class_shift = 1
                class_map = step_class_inds
            elif opts.dataset == "voc":
                n_classes = 21
                skip_inds = [i for i in range(21) if i not in step_class_inds]
                class_shift = 0
                class_map = list(range(21))
            else:
                raise ValueError(f"Dataset {opts.dataset} not supported! Choose between 'coco-voc' and 'voc'.")
            
            for i, pseudolabel in enumerate(pseudolabels):
                pseudolabel_1h = np.zeros((n_classes), dtype=np.uint8)
                for present_class in np.unique(pseudolabel):
                    if present_class not in skip_inds:
                        pseudolabel_1h[class_map[present_class - class_shift] - class_shift] = 1
                # collect classes that were in the image prompt and found by the pseudolabeler
                all_found_classes_str = np.array([classes[i+class_shift] for i in np.arange(n_classes-(1-class_shift))[pseudolabel_1h[(1-class_shift):] == 1]])
                identified_all_flag = True
                for cl in prompted_classes[i]:  # iterate over classes that should be present
                    if cl not in all_found_classes_str:
                        identified_all_flag = False
                if identified_all_flag:
                    pseudolabels_1h[gen_names[i].replace(".jpg", ".png")] = pseudolabel_1h
                    pseudolabel = Image.fromarray(pseudolabel)
                    pseudolabel.save(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels/{gen_names[i].replace('.jpg', '.png')}")
                elif last_iter:
                    pseudolabels_1h[gen_names[i].replace(".jpg", ".png")] = pseudolabel_1h
                    pseudolabel = Image.fromarray(pseudolabel)
                    pseudolabel.save(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels/{gen_names[i].replace('.jpg', '.png')}")
                    unidentified += 1
                else:
                    #     print(f"WARNING: Not all classes identified for {img_name}, saving anyway")
                    unidentified += 1
        if distributed_state.is_main_process:
            with open(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/n_unidentified.txt", "a") as f:
                f.write(f"Number of identified images in this iteration: {unidentified} out of {len(image_names)}\n")
            print(f"Unidentified images: {unidentified} out of {len(image_names)}")
            with open(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels_1h.pkl", "wb") as f:
                pickle.dump(pseudolabels_1h, f)
        del pseudo_labeler
        torch.cuda.empty_cache()
    return unidentified


def copy_real(replay_root, metadata, onehots, opts):
    image_names = list(metadata.keys())
    pseudolabels_1h = dict()
    if opts.dataset == "coco-voc":
        real_img_path = f"data/coco/images/train2017"
        real_img_suffix = ".jpg"
        real_lbl_path = f"data/coco/annotations/train2017"
        real_lbl_suffix = ".png"
    elif opts.dataset == "voc":
        real_img_path = f"data/voc"
        real_img_suffix = ""
        real_lbl_path = f"data/voc"
        real_lbl_suffix = ".png"
        onehots = np.insert(onehots, 0, 0, axis=1)  # add background
    if distributed_state.is_main_process:
            pbar = enumerate(tqdm(image_names, desc="copying"))
    else:
        pbar = enumerate(image_names)
    for i, img_name in pbar:
        os.system(f"cp {real_img_path}/{img_name}{real_img_suffix} {replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/images/{metadata[img_name]['gen_name']}")
        os.system(f"cp {real_lbl_path}/{img_name.replace('JPEGImages', 'SegmentationClassAug').replace('.jpg', '')}{real_lbl_suffix} {replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels/{metadata[img_name]['gen_name'].replace('.jpg', '.png')}")
        pseudolabels_1h[metadata[img_name]["gen_name"].replace(".jpg", ".png")] = onehots[i]
    with open(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels_1h.pkl", "wb") as f:
        pickle.dump(pseudolabels_1h, f)


def post_processing(replay_root, metadata, step_classes, opts):
    for k, v in metadata.items():
        v["classes"] = v["classes"].tolist()
    with open(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/metadata.json", "w") as f:
        json.dump(metadata, f)
    # make grid
    class_to_imgs = {cl: [] for cl in step_classes}
    for d in list(metadata.values()):
        for cl in d["classes"]:
            class_to_imgs[cl].append((d["real_name"], d["gen_name"]))
    non_empty_classes = [cl for cl in step_classes if len(class_to_imgs[cl]) > 0]
    # print(len(step_classes) - len(non_empty_classes), "classes had no images generated.")
    # print(len(non_empty_classes), "classes had images generated.")
    if opts.dataset == "coco-voc":
        real_path = f"data/coco/images/train2017"
        real_suffix = ".jpg"
        if len(non_empty_classes) != 60:
            if distributed_state.is_main_process:
                print(f"WARNING: Normally images should have been created for 60 classes and not created for 31 classes in the list, the real values this time are:, {len(non_empty_classes)} (!= 60), and, {len(step_classes) - len(non_empty_classes)} (!= 31).")
    elif opts.dataset == "voc":
        real_path = f"data/voc"
        real_suffix = ""
        if len(non_empty_classes) != len(step_classes):
            if distributed_state.is_main_process:
                print(f"WARNING: Images found only for {len(non_empty_classes)} out of {len(step_classes)} classes.")
    fig, axes = plt.subplots(2 * len(non_empty_classes), 2, figsize=(10, 10 * len(non_empty_classes)))
    for i, cl in enumerate(non_empty_classes):
        for j in range(2):
            real_img = Image.open(f"{real_path}/{class_to_imgs[cl][j][0]}{real_suffix}").convert("RGB")
            axes[2 * i + j, 0].imshow(real_img)
            axes[2 * i + j, 0].set_title(f"{class_to_imgs[cl][j][0]} ({cl})")
            axes[2 * i + j, 0].set_axis_off()
            gen_img = Image.open(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/images/{class_to_imgs[cl][j][1]}").convert("RGB")
            axes[2 * i + j, 1].imshow(gen_img)
            axes[2 * i + j, 1].set_title(f"{class_to_imgs[cl][j][1]}")
            axes[2 * i + j, 1].set_axis_off()
    plt.tight_layout()
    plt.savefig(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/grid.png")
    plt.close(fig)


if __name__ == "__main__":
    replay_root = sys.argv[1]  # e.g. "replay_data"

    dataset = sys.argv[2]  # "coco-voc" or "voc"
    task = sys.argv[3]  # e.g. "15-5"
    overlap = sys.argv[4] == "overlap"  # "disjoint" or "overlap"
    step = sys.argv[5]  # int or "all"

    resample = int(sys.argv[6])  # int, 0 to deactivate
    shuffle = sys.argv[7] == "shuffle"  # "shuffle" or "noshuffle"
    mode = sys.argv[8]  # "real" to use actual data instead of synthetic, "debug" to test generation with a subset of the data, otherwise full generation is assumed

    gen_model = sys.argv[9]  # "sd21" or "sdxl"
    img_gen_batch_size = int(sys.argv[10])  # e.g. 8
    resolution = int(sys.argv[11])  # e.g. 512
    n_steps = int(sys.argv[12])  # e.g. 50
    prompt_skeleton = sys.argv[13]  # e.g. "a photo of a CLASS"
    neg_prompt = sys.argv[14]  # e.g. "vintage,grayscale,low quality" or "none" to deactivate
    
    base_model_name = sys.argv[15]  # e.g. "Base"
    incr_model_name = sys.argv[16]  # e.g. "Incr_Replay"
    pseudolabel_batch_size = int(sys.argv[17])  # e.g. 32
    
    seed = int(sys.argv[18])  # e.g. 42
    top_k = int(sys.argv[19])  # e.g. 10, the top_k used in LoRA training and select_top_k.py

    # options, sanity checks and warnings
    if step == "all":
        step_list = list(tasks.tasks[dataset][task].keys())[:-1] # exclude last step
    else:
        step_list = [int(step)]
    if neg_prompt == "none":
        neg_prompt = None

    distributed_state = PartialState()
        
    seed_everything(seed)
    
    for current_step in step_list:
        if current_step == 0:
            current_model_name = base_model_name
        else:
            current_model_name = incr_model_name

        opts = wilson_opts_obj(
            dataset=dataset,
            task=task,
            overlap=overlap,
            step=current_step,
            pseudolabeler_name=current_model_name)
        opts.seed = seed
        opts.top_k = top_k
        
        if distributed_state.is_main_process:
            os.makedirs(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/images", exist_ok=True)
            os.makedirs(f"{replay_root}/{opts.dataset}/{opts.task_and_ov}/{opts.step}/pseudolabels", exist_ok=True)

        classes = get_classes(dataset=dataset)
        step_class_inds = tasks.tasks[dataset][task][current_step].copy()
        if 0 in step_class_inds:
            step_class_inds.remove(0)  # no background
        if dataset == "coco-voc":
            step_classes = classes[1:]
        elif dataset == "voc":
            step_classes = [classes[i-1] for i in step_class_inds]

        if distributed_state.is_main_process:
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

        if distributed_state.is_main_process:
            print("Preparing captions...")
        captions, metadata = get_captions(
            replay_root,
            image_names,
            onehot_labels,
            step_classes,
            shuffle,
            prompt_skeleton=prompt_skeleton,
            opts=opts
        )

        if mode == "real":
            if distributed_state.is_main_process:
                print("Copying real images and labels...")
            _, unmasked_onehot_labels = get_image_names_and_onehots(opts)
            copy_real(replay_root, metadata, unmasked_onehot_labels, opts)
        else:
            if distributed_state.is_main_process:
                print("Generating images...")
            generate_images(
                replay_root,
                metadata,
                img_gen_batch_size=img_gen_batch_size,
                seed=seed,
                n_steps=n_steps,
                res=resolution,
                neg_prompt=neg_prompt,
                gen_model=gen_model,
                opts=opts
            )
            if distributed_state.is_main_process:
                print("Generating pseudolabels...")
            unidentified = generate_labels(
                replay_root,
                opts,
                step_class_inds=step_class_inds,
                metadata=metadata,
                pseudolabel_batch_size=pseudolabel_batch_size,
                res=resolution,
                classes=classes,
                gen_model=gen_model,
                last_iter=(resample == 0)
            )
            for i in range(resample):
                if unidentified == 0:
                    if distributed_state.is_main_process:
                        print(f"All images identified after {i} resample iterations, skipping remaining {resample - i} iterations.")
                    break
                if distributed_state.is_main_process:
                    print(f"Resample iteration {i+1}...")
                    print("Regenerating images...")
                regenerate_images(
                    replay_root,
                    metadata,
                    img_gen_batch_size=img_gen_batch_size,
                    seed=seed,
                    n_steps=n_steps,
                    res=resolution,
                    neg_prompt=neg_prompt,
                    gen_model=gen_model,
                    opts=opts,
                    redo_iter=i
                )
                if distributed_state.is_main_process:
                    print("Regenerating pseudolabels...")
                unidentified = regenerate_labels(
                    replay_root,
                    opts,
                    step_class_inds=step_class_inds,
                    metadata=metadata,
                    pseudolabel_batch_size=pseudolabel_batch_size,
                    res=resolution,
                    classes=classes,
                    gen_model=gen_model,
                    last_iter=(i == resample - 1)
                )
        
        distributed_state.wait_for_everyone()
        if distributed_state.is_main_process:
            print("Postprocessing...")
            post_processing(
                replay_root,
                metadata,
                step_classes,
                opts
            )
            print("Done with step", current_step)
    if distributed_state.is_main_process:
        print("Done with all steps!")
    distributed_state.wait_for_everyone()
    distributed_state.destroy_process_group()