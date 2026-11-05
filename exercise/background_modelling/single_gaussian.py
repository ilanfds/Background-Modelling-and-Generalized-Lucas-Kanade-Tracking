# Tutors: Stephanie Käs and Ilya Fradlin
# Learn the RGB-sum mean and ML variance (ddof=0) on indices 0..99; classify from 100.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from background_model import BackgroundModel
from tqdm import tqdm


class SingleGaussian(BackgroundModel):
    def __init__(self, image_path, threshold):
        self.threshold = threshold
        super(SingleGaussian, self).__init__(image_path)

    def _init(self):
        # TODO STUDENT
        images_intensity = np.array([(mpimg.imread(self.file_paths[p]) / 255).sum(axis = 2) for p in range(100)])
        self.mean_img = images_intensity.sum(axis = 0) / 100
        self.var_img = ((images_intensity - self.mean_img)**2).sum(axis=0) / 100

    def update(self, current_frame_id: int) -> None:
        # nothing to do for single Gaussian without updates
        pass

    def apply(self, current_frame_id: int):
        image = mpimg.imread(self.file_paths[current_frame_id]) / 255
        images_intensity = (mpimg.imread(self.file_paths[current_frame_id]) / 255).sum(axis = 2)

        # TODO STUDENT
        self.thresholded_image = np.where(np.abs(images_intensity - self.mean_img) > self.threshold * np.sqrt(self.var_img), True, False )
        self.current_image = image
        self.current_frame_id = current_frame_id

    def plot(self, output_path, save_fig=False):
        super(SingleGaussian, self).plot(output_path, save_fig)
        current_file_name = f"{str(self.current_frame_id).zfill(3)}.jpg"

        fig = plt.figure(figsize=(8, 7))

        ax1 = fig.add_subplot(231)
        ax1.title.set_text(f"Mean Image")
        ax1.imshow(self.mean_img)

        ax2 = fig.add_subplot(233)
        ax2.title.set_text(f"Image {current_file_name}")
        ax2.imshow(self.current_image)

        ax2_3 = fig.add_subplot(235)
        ax2_3.title.set_text(f"Thresholded image (T={round(self.threshold, 2)})")
        ax2_3.imshow(self.thresholded_image)

        ax3 = fig.add_subplot(234)
        ax3.title.set_text(f"Variance Image")
        ax3.imshow(self.var_img)

        overlayed = np.clip(self.current_image +
                            np.repeat(self.thresholded_image[..., np.newaxis], 3, axis=2) * [0.7, 0, 0], 0, 1)

        ax4 = fig.add_subplot(236)
        ax4.title.set_text(f"Overlayed Mask")
        ax4.imshow(overlayed)

        fig.suptitle(self.__class__.__name__)
        if save_fig:
            plt.savefig(f"{output_path}/{str(self.current_frame_id).zfill(self.string_size)}.jpg",
                        dpi=2 * fig.dpi,
                        bbox_inches="tight")
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()

            fig.clf()
            plt.close()
        else:
            plt.show()
