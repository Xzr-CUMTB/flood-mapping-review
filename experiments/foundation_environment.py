"""Use the existing conda geospatial resources from a project-local venv."""
from pathlib import Path
import os, sys


def configure():
    root = Path(__file__).resolve().parents[1]
    share = Path(sys.base_prefix)/'Library/share'
    for key, directory, sentinel in [('PROJ_DATA', share/'proj', 'proj.db'),
                                      ('GDAL_DATA', share/'gdal', 'gdalvrt.xsd')]:
        if not (directory/sentinel).is_file():
            raise FileNotFoundError('Required base-environment resource: '+str(directory/sentinel))
        os.environ[key] = str(directory)
    for key, value in {'HF_HOME': str(root/'models/pretrained/hf_cache'),
                       'TORCH_HOME': str(root/'models/pretrained/torch_cache'),
                       'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
                       'NO_ALBUMENTATIONS_UPDATE': '1', 'WANDB_MODE': 'disabled',
                       'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}.items():
        os.environ.setdefault(key, value)
