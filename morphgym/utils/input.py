import os
import yaml
# from omegaconf import DictConfig
from morphgym.utils.Odict import ODict


def load_yaml(file_path):
    if not os.path.exists(file_path):
        return False
    with open(file_path) as f:
        dict = yaml.load(f, Loader=yaml.FullLoader)
    return dict

def load_cfg(file_path):
    yaml_dict = load_yaml(file_path)
    if not yaml_dict:
        raise FileNotFoundError(f'"{file_path}" file not exits.')
    cfg = ODict(yaml_dict)
    return cfg