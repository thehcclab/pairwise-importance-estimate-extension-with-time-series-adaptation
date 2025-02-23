import logging


def define_root_logger(filename='logfile.txt',filemode='a'):
    logging.root.handlers = []
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)-10s: %(filename)-10s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=filename,
                        filemode=filemode)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True


import yaml, io

def export_model_as_yaml(dictionary, file='model.yaml'):
    with io.open(file, 'w', encoding='utf8') as outfile:
        yaml.dump(dictionary, outfile, default_flow_style=False, allow_unicode=True)

def import_model_from_yaml(file='model.yaml'):
    with open(file, 'r') as stream:
        return yaml.safe_load(stream)
