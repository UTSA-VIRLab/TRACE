# PATH_TO_MIMIC_CXR = '/home/riaz/Desktop/physionet.org/files' #TODO set your own path to MIMIC-CXR-JPG dataset (should point to a folder containing "mimic-cxr-jpg" folder)
# VIS_ROOT = f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.1.0"

# JAVA_HOME = "<PATH_TO_JAVA>/java/jre1.8.0_361" #TODO set your own path to java home, adapt version if necessary
# JAVA_PATH = "PATH_TO_JAVA/java/jre1.8.0_361/bin:"

# CHEXBERT_ENV_PATH = '/home/riaz/miniconda3/envs/radialog/bin/python'
# CHEXBERT_PATH = '/home/riaz/Desktop/RaDialog_v2/chexbert/src' #replace with path to chexbert project in RaDialog folder

# WANDB_ENTITY = " " #TODO set your own wandb entity

PATH_TO_MIMIC_CXR = "/raid/den365/physionet.org/files"
VIS_ROOT = f"{PATH_TO_MIMIC_CXR}/mimic-cxr-jpg/2.1.0"

JAVA_HOME = "/usr/lib/jvm"
JAVA_PATH = "/usr/lib/jvm/bin:"

import sys
CHEXBERT_ENV_PATH = sys.executable

CHEXBERT_PATH = "/raid/den365/RaDialog_v2/chexbert/src"

WANDB_ENTITY = ""