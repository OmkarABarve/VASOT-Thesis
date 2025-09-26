# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================
import os
import datasets
import numpy as np

# =============================================================================
# DATASET DESCRIPTION AND METADATA
# =============================================================================
_DESCRIPTION = """\
GTEA is composed of 50 recorded videos of 25 participants making two different mixed salads. 
The videos are captured by a camera with a top-down view onto the work-surface. 
The participants are provided with recipe steps which are randomly sampled from a statistical recipe model.
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

# =============================================================================
# CONFIGURATION - SET YOUR DATA PATH HERE
# =============================================================================
# 🔧 Change this to your local path
LOCAL_DATA_DIR = r"C:\Users\Admin\Desktop\VASOT-Thesis\gtea"   # This folder must contain "gtea/"

# =============================================================================
# MAIN DATASET CLASS DEFINITION
# =============================================================================
class GTEA(datasets.GeneratorBasedBuilder):

    # =========================================================================
    # VERSION INFORMATION
    # =========================================================================
    VERSION = datasets.Version("1.0.0")

    # =========================================================================
    # CROSS-VALIDATION SPLIT CONFIGURATIONS
    # =========================================================================
    # Define 4 different cross-validation splits
    # Each split uses different subjects for train/test
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

    # =========================================================================
    # DEFAULT CONFIGURATION
    # =========================================================================
    DEFAULT_CONFIG_NAME = "split1"

    # =========================================================================
    # DATASET SCHEMA DEFINITION
    # =========================================================================
    # Define what each example in the dataset contains
    def _info(self):
        features = datasets.Features(
            {
                "video_id": datasets.Value("string"),                    # Video identifier (e.g., "S1_Cheese_C1")
                "video_feature": datasets.Array2D(shape=(None, 2048), dtype="float32"),  # Pre-extracted features (T x 2048)
                "video_label": datasets.Sequence(datasets.Value(dtype="int32")),         # Frame-by-frame action labels
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    # =========================================================================
    # DATA SPLIT GENERATION
    # =========================================================================
    # Define where to find train/test split files and data folders
    def _split_generators(self, dl_manager):
        # Use local path instead of downloading
        data_dir = LOCAL_DATA_DIR
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

    # =========================================================================
    # DATA LOADING AND PROCESSING
    # =========================================================================
    # This function loads individual examples from files
    def _generate_examples(self, filepath, featurefolder, gtfolder, mappingpath):
        # =================================================================
        # LOAD ACTION MAPPING
        # =================================================================
        # Read mapping.txt to convert action names to numbers
        with open(mappingpath, "r") as f:
            actions = f.read().splitlines()
        actions_dict = {}
        for a in actions:
            actions_dict[a.split()[1]] = int(a.split()[0])

        # =================================================================
        # PROCESS EACH VIDEO IN THE SPLIT
        # =================================================================
        with open(filepath, "r") as f:
            lines = f.read().splitlines()
            for key, line in enumerate(lines):
                # Extract video ID (remove .txt extension)
                vid = line[:-4]
                
                # =========================================================
                # LOAD PRE-EXTRACTED FEATURES
                # =========================================================
                featurepath = os.path.join(featurefolder, f"{vid}.npy")
                feature = np.load(featurepath).T  # T x D

                # =========================================================
                # LOAD GROUND TRUTH LABELS
                # =========================================================
                gtpath = os.path.join(gtfolder, line)
                with open(gtpath, "r") as fgt:
                    content = fgt.read().splitlines()

                # =========================================================
                # CREATE LABEL ARRAY
                # =========================================================
                # Match feature length with label length
                label = np.zeros(min(np.shape(feature)[0], len(content)))
                for i in range(len(label)):
                    label[i] = actions_dict[content[i]]

                # =========================================================
                # YIELD EXAMPLE
                # =========================================================
                # Return one complete example with video_id, features, and labels
                yield key, {
                    "video_id": vid,
                    "video_feature": feature,
                    "video_label": label,
                }