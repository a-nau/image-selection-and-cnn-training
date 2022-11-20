import cv2
import numpy as np
from detectron2.engine import DefaultPredictor


def check_image_plain(im):
    flag = check_image_manually(im)
    if flag == "selected":
        mask_convexity_score = compute_mask_convexity_score_rgba(im)
        if mask_convexity_score < 0.95:
            flag = "mask_not_convex"
    return flag


def check_image_manually(im):
    mask_transparency_score = compute_mask_transparency_score(im)
    if mask_transparency_score > 0.1:
        flag = "mask_transparent"
    else:
        flag = "selected"
    return flag


def check_image_with_maskrcnn(im, predictor: DefaultPredictor):
    im_rgb = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2BGR)  # convert RGBA image to RGB
    # visualize_image(im)
    outputs = predictor(im_rgb)

    flag = "selected"  # since we removed background beforehand, we only consider suitable candidates
    num_masks = len(outputs["instances"])
    if num_masks > 1:
        flag = "multiple_masks"
    elif num_masks == 0:
        flag = "no_detection"
    if flag == "selected":
        flag = check_image_plain(im)
    return flag


def compute_mask_transparency_score(im, threshold=0.95):
    """
    Compute percentage of non-zero values, that are below a given threshold. A high value means, that there is many
    transparent points. A low value means, that the mask is almost binary.
    """
    alpha_channel = im[..., 3].copy() / 255  # alpha channel is already a binary mask
    non_zero = np.count_nonzero(alpha_channel)
    if non_zero == 0:
        return 1
    transition = alpha_channel.copy()
    transition[transition > threshold] = 0
    transition_non_zero = np.count_nonzero(transition)
    return transition_non_zero / non_zero


def compute_mask_convexity_score_rgba(im):
    alpha_channel = im[..., 3]  # alpha channel is already a binary mask
    mask = np.zeros(alpha_channel.shape, np.uint8)
    mask_convex = np.zeros(alpha_channel.shape, np.uint8)
    # Find biggest contour
    contours, _ = cv2.findContours(
        alpha_channel, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    max_contour = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]
    cv2.drawContours(mask, [max_contour], 0, 255, -1)  # fill contour
    # Find convex hull thereof
    convex_hull = cv2.convexHull(max_contour.astype(np.float32))  # need float input
    convex_hull = convex_hull.astype(np.int32).reshape(-1, 2)  # fill needs int input
    cv2.drawContours(mask_convex, [convex_hull], 0, 255, -1)  # fill contour
    # Check size differences
    mask_convexity_score = np.sum(mask) / np.sum(mask_convex)
    return mask_convexity_score


def check_if_image_background_is_homogeneous(
    img: np.ndarray, margin=0.02, threshold=50
):
    height, width = img.shape[:2]
    margin_width = max(round(margin * width), 1)
    margin_height = max(round(margin * height), 1)
    colors = np.vstack(
        (
            img[:margin_height, ...].reshape(-1, 3),
            img[-margin_height:, ...].reshape(-1, 3),
            img[:, :margin_width, :].reshape(-1, 3),
            img[:, -margin_width:, :].reshape(-1, 3),
        )
    )
    colors_normalized = colors - (0.5 / 255)
    variance = np.var(colors_normalized, axis=0)
    mean_variance = np.mean(variance)
    print(mean_variance)
    # print(f"Mean: {mean_variance:.2f}, Var: {variance}")
    if mean_variance < threshold:
        return True
    return False
