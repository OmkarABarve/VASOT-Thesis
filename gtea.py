import os
import datasets

import numpy as np

_DESCRIPTION = """\
GTEA is composed of 50 recorded videos of 25 participants making two different mixed salads. The videos are captured by a camera with a top-down view onto the work-surface. The participants are provided with recipe steps which are randomly sampled from a statistical recipe model.
"""

_CITATION = """\
@inproceedings{stein2013combining,
title={Combining embedded accelerometers with computer vision for recognizing food preparation activities},
author={Stein, Sebastian and McKenna, Stephen J},
booktitle={Proceedings of the 2013 ACM international joint conference on Pervasive and ubiquitous computing},
pages={729--738},
year={2013}
}
"""

_HOMEPAGE = ""

_LICENSE = "xxx"

_URLS = {"full": "https://huggingface.co/datasets/dinggd/gtea/resolve/main/gtea.zip"}


class GTEA(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="split1", version=VERSION, description="Cross Validation Split1"
        ),
        datasets.BuilderConfig(
            name="split2", version=VERSION, description="Cross Validation Split2"
        ),
        datasets.BuilderConfig(
            name="split3", version=VERSION, description="Cross Validation Split3"
        ),
        datasets.BuilderConfig(
            name="split4", version=VERSION, description="Cross Validation Split4"
        ),
    ]

    DEFAULT_CONFIG_NAME = "1"

    def _info(self):
        features = datasets.Features(
            {
                "video_id": datasets.Value("string"),
                "video_feature": datasets.Array2D(shape=(None, 2048), dtype="float32"),
                "video_label": datasets.Sequence(datasets.Value(dtype="int32")),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = _URLS
        data_dir = dl_manager.download_and_extract(urls_to_download)["full"]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, f"gtea/splits/train.{self.config.name}.bundle"
                    ),
                    "featurefolder": os.path.join(data_dir, "gtea/features"),
                    "gtfolder": os.path.join(data_dir, "gtea/groundTruth"),
                    "mappingpath": os.path.join(data_dir, "gtea/mapping.txt"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(
                        data_dir, f"gtea/splits/test.{self.config.name}.bundle"
                    ),
                    "featurefolder": os.path.join(data_dir, "gtea/features"),
                    "gtfolder": os.path.join(data_dir, "gtea/groundTruth"),
                    "mappingpath": os.path.join(data_dir, "gtea/mapping.txt"),
                },
            ),
        ]

    def _generate_examples(self, filepath, featurefolder, gtfolder, mappingpath):
        with open(mappingpath, "r") as f:
            actions = f.read().splitlines()
        actions_dict = {}
        for a in actions:
            actions_dict[a.split()[1]] = int(a.split()[0])

        with open(filepath, "r") as f:
            lines = f.read().splitlines()
            for key, line in enumerate(lines):
                vid = line[:-4]
                featurepath = os.path.join(featurefolder, f"{vid}.npy")
                gtpath = os.path.join(gtfolder, line)
                feature = np.load(featurepath).T  # T x D
                with open(gtpath, "r") as f:
                    content = f.read().splitlines()

                label = np.zeros(min(np.shape(feature)[1], len(content)))
                for i in range(len(label)):
                    label[i] = actions_dict[content[i]]
                yield key, {
                    "video_id": vid,
                    "video_feature": feature,
                    "video_label": label,
                }
