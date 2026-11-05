# Tutors: Stephanie Käs and Ilya Fradlin
# delay is measured in input frames. Call update(t) before apply(t). Clip negative reference indices to zero.
import matplotlib.image as mpimg
from background_substraction import BackgroundSubtraction


class FrameDifferencing(BackgroundSubtraction):
    def __init__(self, image_path, threshold, delay):
        if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0:
            raise ValueError("delay must be a non-negative integer")
        self.delay = delay
        super(FrameDifferencing, self).__init__(image_path, threshold)

    def _init(self):
        # TODO STUDENT
        self.background_model = mpimg.imread(self.file_paths[0]) / 255
        self.background_file_name = f"{str(0).zfill(3)}.jpg"

    def update(self, current_frame_id: int) -> None:
        # TODO STUDENT
        self.background_id = current_frame_id - self.delay if current_frame_id - self.delay > 0 else 0
        self.background_file_name = f"{str(self.background_id).zfill(3)}.jpg"
        self.background_model = mpimg.imread(self.file_paths[self.background_id]) / 255
