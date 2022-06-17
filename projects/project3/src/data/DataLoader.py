from __future__ import annotations

from tensorflow_datasets.core.load import is_full_name

""" ISIC DATASET LOADER """

from projects.utils import get_project3_root

proot = get_project3_root()

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

class ISICDataLoader(tfds.core.GeneratorBasedBuilder):
    """ ISIC Dataset"""
    VERSION = tfds.core.Version("3.2.0")

    def _info(self):
        return

    def _split_generators(self, dl_manager):
        pass

    def _generate_examples(self, **kwargs):
        pass

if __name__ == "__main__":
    ISICDataLoader()
