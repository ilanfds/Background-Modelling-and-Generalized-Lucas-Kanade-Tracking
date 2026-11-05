# Tutors: Stephanie Käs and Ilya Fradlin
# Support code for Exercise 1; algorithms and interfaces follow the supplied Python sources.
from abc import ABC, abstractmethod
from glob import glob
import os


class BackgroundModel(ABC):
    def __init__(self, image_path):
        self.file_paths = sorted(glob(f"{image_path}/*.jpg"))
        self.length = len(self.file_paths)
        if self.length == 0:
            raise ValueError("No JPEG frames found in the input directory")
        self.string_size = len(str(self.length))

        self._init()

    def __len__(self):
        return self.length

    @abstractmethod
    def _init(self):
        pass

    @abstractmethod
    def update(self, current_frame_id):
        pass

    @abstractmethod
    def apply(self, current_frame_id):
        pass

    @abstractmethod
    def plot(self, output_path, save_fig):
        if save_fig:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
