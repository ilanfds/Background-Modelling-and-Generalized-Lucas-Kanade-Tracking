# Tutors: Stephanie Käs and Ilya Fradlin
# Compare the present with both past and future. The intersection belongs to the middle time; future frames are required.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from background_model import BackgroundModel


class ThreeFrameDifferencing(BackgroundModel):
    def __init__(self, image_path, threshold, delay):
        self.threshold = threshold
        if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
            raise ValueError("delay must be a non-negative integer")
        self.delay = delay
        super(ThreeFrameDifferencing, self).__init__(image_path)

    def _init(self):
        # TODO STUDENT
        pass

    def update(self, current_frame_id: int) -> None:
        # TODO Student
        self.past_background_file_name = f"{str(current_frame_id - self.delay).zfill(3)}.jpg" if current_frame_id - self.delay > 0 else f"{str(0).zfill(3)}.jpg"
        self.future_background_file_name = f"{str(current_frame_id + self.delay).zfill(3)}.jpg" if current_frame_id + self.delay < self.length else f"{str(self.length - 1).zfill(3)}.jpg"

        self.past_background_model = mpimg.imread(self.file_paths[current_frame_id - self.delay]) / 255 if current_frame_id - self.delay > 0 else mpimg.imread(self.file_paths[0]) / 255
        self.future_background_model = mpimg.imread(self.file_paths[current_frame_id + self.delay]) / 255 if current_frame_id + self.delay < self.length else mpimg.imread(self.file_paths[self.length - 1]) / 255

    def apply(self, current_frame_id: int):
        image = mpimg.imread(self.file_paths[current_frame_id]) / 255

        # TODO STUDENT
        self.future_thresh_image = np.where(np.abs(image - self.future_background_model).sum(axis = 2) >= self.threshold,True,False)
        self.past_thresh_image = np.where(np.abs(image - self.past_background_model).sum(axis = 2) >= self.threshold,True,False)
        self.thresholded_image = np.logical_and(self.past_thresh_image, self.future_thresh_image)
        self.current_image = image
        self.current_frame_id = current_frame_id

    def plot(self, output_path, save_fig=False):
        current_file_name = f"{str(self.current_frame_id).zfill(3)}.jpg"

        fig = plt.figure(figsize=(8, 7))

        ax1 = fig.add_subplot(331)
        ax1.title.set_text(f"Image {self.past_background_file_name} for past")
        ax1.imshow(self.past_background_model)

        ax2 = fig.add_subplot(332)
        ax2.title.set_text(f"Image {current_file_name}")
        ax2.imshow(self.current_image)

        ax2_1 = fig.add_subplot(333)
        ax2_1.title.set_text(f"Image {self.future_background_file_name} for future")
        ax2_1.imshow(self.future_background_model)

        ax2_2 = fig.add_subplot(334)
        ax2_2.title.set_text(f"past <-> current")
        ax2_2.imshow(self.past_thresh_image)

        ax3 = fig.add_subplot(335)
        ax3.title.set_text(f"threshold image AND")
        ax3.imshow(self.thresholded_image)

        ax4 = fig.add_subplot(336)
        ax4.title.set_text(f"current <-> future")
        ax4.imshow(self.future_thresh_image)

        overlayed = np.clip(self.current_image +
                            np.repeat(self.thresholded_image[..., np.newaxis], 3, axis=2) * [0.7, 0, 0], 0, 1)

        ax4 = fig.add_subplot(338)
        ax4.title.set_text(f"Overlayed Mask")
        ax4.imshow(overlayed)

        fig.suptitle(self.__class__.__name__)
        if save_fig:
            plt.savefig(f"{output_path}/{str(self.current_frame_id).zfill(self.string_size)}.jpg",
                        dpi=2 * fig.dpi,
                        bbox_inches="tight")
            ax1.cla()
            ax2.cla()
            ax2_1.cla()
            ax3.cla()
            ax4.cla()

            fig.clf()
            plt.close()
        else:
            plt.show()
