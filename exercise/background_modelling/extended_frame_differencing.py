# Tutors: Stephanie Käs and Ilya Fradlin
# history is a fading motion trace: subtract 10/255 per processed frame; a new detection resets its value to one.
import numpy as np
import matplotlib.pyplot as plt
from frame_differencing import FrameDifferencing


class ExtendedFrameDifferencing(FrameDifferencing):
    def __init__(self, image_path, threshold, delay):
        self.history = None
        super(ExtendedFrameDifferencing, self).__init__(image_path, threshold, delay)

    def _init(self):
        self.update(0)

    def apply(self, current_frame_id: int):
        super(ExtendedFrameDifferencing, self).apply(current_frame_id)

        #initialize
        if self.history is None:
            self.history = np.zeros(self.background_model.shape[:-1])

        #dont allow the value to be low than 0
        tmp = np.maximum(self.history - 10/255, 0)
        #Pixels antigos vao se "apagando" se n for detectado naquele pixel novamente, vao ficando de cor diferente dando para perceber o movimento de um objeto ate outro ponto pela diferenca de cor
        self.history = np.maximum(self.thresholded_image.astype(int), tmp)

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

        ax4 = fig.add_subplot(224)
        ax4.title.set_text(f"History")
        ax4.imshow(self.history)

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
