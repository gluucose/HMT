import json
from pathlib import Path

from ruamel.yaml import YAML

from utils import JsonObject
from src.trainer_model import TrainerModel


def test(cfg):
    pass


if __name__ == '__main__':
    yaml_path = {'model': r'../config/model.yaml',}
    
    yaml = YAML(typ='safe')
    cfg_json = {}
    cfg = {}
    for k, v in yaml_path.items():
        cfg_json[k] = json.dumps(yaml.load(Path(v).open('r')), indent=4)
        cfg[k] = json.loads(cfg_json[k], object_hook=JsonObject)

    TrainerModel(cfg['model'], cfg_json['model']).run()
    
    # test(cfg)
