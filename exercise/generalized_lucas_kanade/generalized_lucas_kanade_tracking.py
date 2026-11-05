# Tutors: Stephanie Käs and Ilya Fradlin
# Coordinates are (x,y), array indexing is [y,x]. Parameters are (tx,ty,theta), with angles in radians. IC pairs error I(W)-T with A @ inverse(delta_A).
import cv2
import matplotlib.pyplot as plt
import math
import torch
from glob import glob
from functools import lru_cache
from tqdm import tqdm
import argparse


def warp_image(img, A, point_coords, img_size):
    """
    Warp template point coordinates onto the image using the affine transformation A.
    Use bilinear interpolation to obtain image intensities (subpixel accuracy is important!)
    """
    warped_points = torch.einsum('ij,klj -> kli', A, point_coords)
    return torch.nn.functional.grid_sample(img.unsqueeze(0).unsqueeze(0),
                                           (2 * (warped_points[:, :, :2] / img_size) - 1).unsqueeze(0),  # [-1,1] range
                                           align_corners=True).squeeze()


@lru_cache(maxsize=None)
def get_differential_filter():
    """
    get differential filter in both x and y direction
    """

    # === STUDENT SOLUTION GOES HERE ===
    # Central derivative must include the factor 1/2.
    filter_x = None
    filter_y = None
    # ==================================
    return torch.stack((filter_x, filter_y)).unsqueeze(1)


@lru_cache(maxsize=None)
def jacobian_at_p0(x_shape, y_shape):
    """
    Evaluate the Jacobian del_W / del_p at (x;p=0).
    See slide 49 Slide Set 3.
    Think about how the Jacobian looks like for translation + rotation warping!
    """

    # === STUDENT SOLUTION GOES HERE ===
    jacob = None
    # ==================================

    return jacob


def get_affine_transform_from_tr(p):
    """
    calculate affine transformation matrix from translation+rotation parameters p
    """

    # === STUDENT SOLUTION GOES HERE ===
    return None
    # ==================================


def align_image(template, target, A, point_coords, img_size):
    # Step 3 -> Evaluate nabla_T
    img_filter = get_differential_filter()
    delta_template = None  # TODO (keep the same image dimension using padding!)

    # Step 4 -> Evaluate the Jacobian del_W / del_p at (x;p=0)
    jacobian = jacobian_at_p0(template.shape[1], template.shape[0])

    # Step 5 -> Compute the steepest descent images nabla_T * del_W / del_p
    steepest_descent_images = None  # TODO

    # Step 6.1 -> Compute the Hessian matrix
    hessian = None  # TODO
    # Step 6.2 -> Compute the inverse of the Hessian matrix
    hessian_inv = None  # TODO Check out PyTorch documentation on how to calculate the inverse

    # Refining warp function (here translation+rotation)
    refined_A = A
    error_norms = []
    for num_iterations in range(args.max_refine_iters):
        # Step 1 -> Warp template with current warp function
        target_warped = warp_image(img=target,
                                   A=refined_A,
                                   point_coords=point_coords,
                                   img_size=img_size)

        # Step 2 -> Compute error image, i.e. warped_template - template
        Ierr = None

        # Optional: Keep track of error norm
        # Useful for early stopping of refinement if changes are marginal
        # Alternatively: check ||del_p|| -> 0
        error_norms.append(None)  # TODO

        # Step 7 -> SD Parameter Updates
        sd_param_update = None  # TODO

        # Step 8 -> Compute delta_p
        delta_p = None  # TODO
        refined_A = None  # TODO

        if error_norm < args.early_stopping:
            break

    return refined_A, error_norms


def track_multi_frames(original_template, initial_warp, frame_list):
    A_list = []
    template = original_template

    # calculate template point coordinates (homogenous!)
    xx, yy = torch.meshgrid(torch.arange(template.shape[0]), torch.arange(template.shape[1]))
    point_coords = torch.cat((yy.unsqueeze(-1), xx.unsqueeze(-1)), dim=-1)
    ones = torch.ones((point_coords.shape[0], point_coords.shape[1], 1))
    point_coords = torch.cat((point_coords, ones), dim=-1)

    img_size = torch.tensor([frame_list[0].shape[1] - 1, frame_list[0].shape[0] - 1])

    A = initial_warp

    for i, target in enumerate(tqdm(frame_list, desc="align template to next frame")):
        A, errors = align_image(template, target, A,
                                point_coords,
                                img_size)
        template = warp_image(target, A, point_coords, img_size)
        A_list.append(A)

    return A_list


def visualize_track_multi_frames(template, img_list, A_list, output_path):
    bbox_list = []
    for A in A_list:
        template_boundaries = torch.hstack((torch.tensor([[0, 0],
                                                          [template.shape[1], 0],
                                                          [template.shape[1], template.shape[0]],
                                                          [0, template.shape[0]],
                                                          [0, 0]]),
                                            torch.ones((5, 1))))

        boundary_points = template_boundaries @ A[:2, :].T
        bbox_list.append(boundary_points)

    for i in tqdm(range(len(img_list)), desc="write results to disk"):
        plt.imshow(img_list[i], cmap='gray')
        plt.plot(bbox_list[i][:, 0], bbox_list[i][:, 1], 'r')
        plt.savefig(f'{output_path}/{i:03d}.png')
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generalized Lucas-Kanade Template Tracking")
    parser.add_argument("--input_path", help="path to the input video sequence", default="videos/postcard")
    parser.add_argument("--template_path", help="path to the template to track", default="templates/postcard_template.png")
    parser.add_argument("--output_path", help="output directory", default="output")
    parser.add_argument("--max_refine_iters", help="how many maximal warp refinement steps", type=int, default=80)
    parser.add_argument("--early_stopping", help="early stopping error", type=float, default=0.001)
    parser.add_argument("--initial_warp", help="early stopping error", type=float, nargs="+", default=[1., 0., 149.,
                                                                                           0., 1., 148.,
                                                                                           0., 0., 1.])

    args = parser.parse_args()
    if args.max_refine_iters < 1 or not math.isfinite(args.early_stopping) or args.early_stopping < 0:
        parser.error("Require positive iteration count and non-negative finite stopping threshold")
    if len(args.initial_warp) != 9:
        parser.error("initial_warp requires nine numbers")

    # map point coordinates from template onto image
    initial_warp = torch.tensor(args.initial_warp).view(3, 3)

    template_array = cv2.imread(args.template_path, 0)
    if template_array is None:
        parser.error("Cannot read template image")
    template = torch.from_numpy(template_array).float()
    frame_list = []
    for frame_path in tqdm(sorted(glob(f"{args.input_path}/*.png")), desc="read in targets"):
        frame = cv2.imread(frame_path, 0)  # read as grey scale image
        frame_list.append(torch.from_numpy(frame).float())

    if not frame_list:
        parser.error("No PNG target frames found")
    import os
    os.makedirs(args.output_path, exist_ok=True)
    A_list = track_multi_frames(template, initial_warp, frame_list)
    visualize_track_multi_frames(template, frame_list, A_list, args.output_path)
