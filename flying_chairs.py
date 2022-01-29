import cv2
import random

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm.cli import tqdm
from fire import Fire

from mmflow.datasets.utils.flow_io import (read_flow,
                                           visualize_flow)
from mmflow.core.evaluation.metrics import (end_point_error,
                                            end_point_error_map)
from mmflow.apis import (inference_model,
                         init_model)


def get_flying_chairs_dict(data_folder):
    filenames = list(sorted(data_folder.glob("*")))
    fcd = {}
    for fn in filenames:
        ext = fn.suffix
        assert ext in [".ppm", ".flo"], f"{ext} {fn}"
        pair_idx, item_type = fn.stem.split("_")

        if pair_idx not in fcd:
            fcd[pair_idx] = {}

        fcd[pair_idx][item_type] = fn

    return fcd


def cv2_imread(fn):
    img = cv2.imread(str(fn))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_patch_location(img, size=(50, 50)):
    img_h, img_w, img_c = img.shape
    patch_h, patch_w = size

    x = np.random.randint(0, img_w - patch_w)
    y = np.random.randint(0, img_h - patch_h)
    location = (x, y)
    return location


def add_random_patch(img, size=(50, 50), location=None):
    img_h, img_w, img_c = img.shape
    patch_h, patch_w = size

    if location is None:
        location = get_patch_location(img, size)

    img = img.copy()
    img[
        location[1]: location[1]+patch_h,
        location[0]: location[0]+patch_w, :
    ] = np.random.randint(0, 255, size=(patch_h, patch_w, img_c))

    return img


def create_circular_mask(h, w):
    center = [int(w/2), int(h/2)]
    radius = min(center[0], center[1], w-center[0], h-center[1])-2
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask


def add_adversarial_patch(img, size=(50, 50), location=None):
    img_h, img_w, img_c = img.shape
    patch_h, patch_w = size

    if location is None:
        location = get_patch_location(img, size)

    patch_img = cv2_imread("universal-patch.png")
    patch_img = cv2.resize(patch_img, size)
    patch_mask = create_circular_mask(*size)

    img = img.copy()
    patch_bg = img[location[1]: location[1]+patch_h, location[0]: location[0]+patch_w, :]
    patch_bg[patch_mask] = patch_img[patch_mask]

    img[
        location[1]: location[1]+patch_h,
        location[0]: location[0]+patch_w, :
    ] = patch_bg

    return img


def main(
    n_selected=10,
    patch_size=(50, 50),
    random_state=42,
    create_plot=False,
    model_name="raft",
    patch_mode="adversarial",
):
    random.seed(random_state)
    np.random.seed(random_state)

    data_folder = Path("data/flying-chairs/FlyingChairs_release/data")
    fcd = get_flying_chairs_dict(data_folder)
    fcd_selected = random.sample(list(fcd.items()), k=n_selected)

    assert patch_mode in ["random", "adversarial"], f"No such patch mode: {patch_mode}"
    if patch_mode == "random":
        patched_fn = add_random_patch
    elif patch_mode == "adversarial":
        patched_fn = add_adversarial_patch

    assert model_name in ["flownet", "pwc", "raft"], f"No such model: {model_name}"
    if model_name == "flownet":
        config_file = "mmflow-configs/flownetc_8x1_slong_flyingchairs_384x448.py"
        checkpoint_file = "mmflow-models/flownetc_8x1_slong_flyingchairs_384x448.pth"
    if model_name == "pwc":
        # config_file = "/mmflow/configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py"
        # checkpoint_file = "mmflow-models/pwcnet_ft_4x1_300k_sintel_final_384x768.pth"
        config_file = "/mmflow/configs/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448.py"
        checkpoint_file = "mmflow-models/pwcnet_8x1_slong_flyingchairs_384x448.pth"
    if model_name == "raft":
        config_file = "mmflow-configs/raft_8x2_100k_flyingchairs_368x496.py"
        checkpoint_file = "mmflow-models/raft_8x2_100k_flyingchairs.pth"

    device = "cuda:0"
    model = init_model(config_file, checkpoint_file, device=device)


    epe_scores = []
    epe_patched_scores = []

    if create_plot:
        plt.figure(figsize=(40, 50))

    for idx, (pair_idx, pair) in tqdm(enumerate(fcd_selected), total=n_selected):

        img1 = cv2_imread(pair["img1"])
        img2 = cv2_imread(pair["img2"])
        flow = read_flow(pair["flow"])

        pred_flow = inference_model(model, img1, img2)

        epe_score = end_point_error_map(pred_flow, flow).mean()
        epe_scores.append(epe_score)


        patch_location = get_patch_location(img1, patch_size)
        patch_img1 = patched_fn(img1, patch_size, patch_location)
        patch_img2 = patched_fn(img2, patch_size, patch_location)

        pred_patched_flow = inference_model(model, patch_img1, patch_img2)

        epe_patched_score = end_point_error_map(pred_patched_flow, flow).mean()
        epe_patched_scores.append(epe_patched_score)

        if create_plot:
            imgs_cat = np.hstack([
                img1, img2,
                visualize_flow(flow), visualize_flow(pred_flow),
                patch_img1, patch_img2,
                visualize_flow(pred_patched_flow),
            ])

            plt.subplot(n_selected, 1, idx+1)
            plt.title(f"epe score: {epe_score:.4f}, epe patched score: {epe_patched_score:.4f}")
            plt.imshow(imgs_cat)

    if create_plot:
        plt.savefig("sample-flying-chairs", bbox_inches="tight")

    mean_epe = np.mean(epe_scores)
    mean_epe_patched = np.mean(epe_patched_scores)
    rel_epe_decrease = ((mean_epe_patched / mean_epe) - 1.0) * 100.0
    print(f"mean epe: {mean_epe:.4f}")
    print(f"mean epe patched: {mean_epe_patched:.4f}")
    print(f"relative epe decrease: {rel_epe_decrease:.4f}")


if __name__ == "__main__":
    Fire(main)

