# Tutors: Stephanie Käs and Ilya Fradlin
# Classify first, then update. The variance residual uses the ALREADY updated mean. alpha=0 freezes the model.
import numpy as np
import matplotlib.image as mpimg
from single_gaussian import SingleGaussian


class SingleGaussianEMAF(SingleGaussian):
    def __init__(self, image_path: str, threshold: float, alpha: float):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must lie in [0, 1]")
        self.alpha = alpha
        super(SingleGaussianEMAF, self).__init__(image_path, threshold)

    def update(self, current_frame_id: int) -> None:
        image = mpimg.imread(self.file_paths[current_frame_id]) / 255
        intensity_image = np.sum(image, axis=2)

        # TODO STUDENT
        # learn new mean and variance
        self.mean_img = (self.mean_img * (1 - self.alpha)) + (self.alpha * intensity_image)
        self.var_img = ((1- self.alpha)*self.var_img) + (self.alpha)*((intensity_image-self.mean_img)**2)
