# Tutors: Stephanie Käs and Ilya Fradlin
# Support code for Exercise 1; algorithms and interfaces follow the supplied Python sources.
import os
import argparse
from tqdm import tqdm
from background_substraction import BackgroundSubtraction
from frame_differencing import FrameDifferencing
from three_frame_differencing import ThreeFrameDifferencing
from extended_frame_differencing import ExtendedFrameDifferencing
from single_gaussian import SingleGaussian
from single_gaussian_emaf import SingleGaussianEMAF


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default="data/sequence-01",
                        help="root directory of surveillance sequence")
    parser.add_argument("--output_path", type=str, default="output",
                        help="output directory")
    parser.add_argument("--threshold", type=float, default=None,
                       help="threshold for ...")
    parser.add_argument("--delay", type=int, default=10,
                       help="delay for q2")
    parser.add_argument("--alpha", type=float, default=0.001,
                       help="learning rate for Single Gaussian EMAF")
    parser.add_argument("--stride", type=int, default=50,
                       help="only output a fraction of frames (plt.savefig(.) takes time...)")
    parser.add_argument('--save_fig', dest='save_fig', action='store_true', help="create video")
    parser.set_defaults(save_fig=False)
    parser.add_argument('--task', type=int, default=1,
                        help='choose the exercise task')

    args = parser.parse_args()
    if args.task not in range(1, 7):
        parser.error("task must be one of 1, 2, 3, 4, 5, 6")
    if args.stride <= 0:
        parser.error("stride must be positive")
    if args.delay < 0:
        parser.error("delay must be non-negative")
    if not 0 <= args.alpha <= 1:
        parser.error("alpha must lie in [0, 1]")
    if args.threshold is None:
        args.threshold = 0.7 if args.task <= 4 else 2.0
    if not __import__('math').isfinite(args.threshold) or args.threshold < 0:
        parser.error("threshold must be finite and non-negative")

    if args.task == 1:
        model = BackgroundSubtraction(args.image_path, args.threshold)
    elif args.task == 2:
        model = FrameDifferencing(args.image_path, args.threshold, args.delay)
    elif args.task == 3:
        model = ThreeFrameDifferencing(args.image_path, args.threshold, args.delay)
    elif args.task == 4:
        model = ExtendedFrameDifferencing(args.image_path, args.threshold, args.delay)
    elif args.task == 5:
        model = SingleGaussian(args.image_path, args.threshold)
    elif args.task == 6:
        model = SingleGaussianEMAF(args.image_path, args.threshold, args.alpha)

    output_dir = f"{args.output_path}/{model.__class__.__name__}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.task in [1,2,3,4]:
        for i in tqdm(range(len(model)), desc="Process images"):
            model.update(i)
            model.apply(i)

            if i % args.stride == 0:
                model.plot(output_dir, args.save_fig)
    else:
        for i in tqdm(range(100, len(model)), desc="Process images"):
            model.apply(i)
            model.update(i)

            if i % args.stride == 0:
                model.plot(output_dir, args.save_fig)

    # ffmpeg -r 1 -i %03d.jpg -vcodec libx264 -y -an -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2"  video.mp4


if __name__ == '__main__':
    main()
