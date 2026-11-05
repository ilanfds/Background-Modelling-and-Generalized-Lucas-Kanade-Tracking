# Tutors: Stephanie Käs and Ilya Fradlin
# Convert RGB to floating point in [0,1] first. Sum absolute channel differences, giving [0,3]. True denotes foreground.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from background_model import BackgroundModel


class BackgroundSubtraction(BackgroundModel):
    def __init__(self, image_path, threshold):
        self.threshold = threshold
        super(BackgroundSubtraction, self).__init__(image_path)

    def _init(self):
        self.background_model = mpimg.imread(self.file_paths[0]) / 255
        self.background_file_name = f"{str(0).zfill(3)}.jpg"

    def update(self, current_frame_id: int) -> None:
        # nothing to do for simple background subtraction
        pass

    def apply(self, current_frame_id: int):
        image = mpimg.imread(self.file_paths[current_frame_id]) / 255

        #TODO STUDENT
        self.thresholded_image = np.where(np.abs(self.background_model - image).sum(axis = 2) >= self.threshold,True,False)
        self.current_image = image
        self.current_frame_id = current_frame_id


    def plot(self, output_path, save_fig=False):
        current_file_name = f"{str(self.current_frame_id).zfill(3)}.jpg"

        fig = plt.figure(figsize=(8, 7))

        ax1 = fig.add_subplot(221)
        ax1.title.set_text(f"Image {self.background_file_name} for Background Model")
        ax1.imshow(self.background_model)

        ax2 = fig.add_subplot(222)
        ax2.title.set_text(f"Image {current_file_name}")
        ax2.imshow(self.current_image)

        ax3 = fig.add_subplot(223)
        ax3.title.set_text(f"Thresholded image (T={round(self.threshold, 2)})")
        ax3.imshow(self.thresholded_image)

        overlayed = np.clip(self.current_image +
                            np.repeat(self.thresholded_image[..., np.newaxis], 3, axis=2)* [0.7, 0, 0], 0, 1)

        ax4 = fig.add_subplot(224)
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
