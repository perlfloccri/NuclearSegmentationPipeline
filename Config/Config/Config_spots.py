import json
import argparse
import os

class UNETSettings:
    config_path = r'D:\DeepLearning\Kaggle\Config\config_spots.json'

    network_info = {
        "max_epochs": 400,
        "dataset": "tisquant",
        "net_description": None,
        "results_folder": "D:\\DeepLearning\\Pipelines\\SegmentationPaper\\results",
        "dataset_dirs_train": None,
        "dataset_dirs_val": None,
        "dataset_dirs_test": None,
        "results_folder": None,
        "netinfo": None,
        "scaled": None,
        "overlap_train": None,
        "overlap_test": None
    }

    def __init__(self):
        try:
            self.readValues()
        except:
            e=1

    def writeValues(self, **kwargs):
        self.network_info.update(kwargs)
        with open(self.config_path, 'w') as f:
            json.dump(self.network_info, f)

    def readValues(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        self.network_info.update(config)

def main():

    settings = UNETSettings()
    try:
        parser = argparse.ArgumentParser(description='Set config for actual network')
        parser.add_argument('--dataset', help='select dataset for training.', default="tisquant")
        parser.add_argument('--net_description', help='select dataset for training.', default=None)
        parser.add_argument('--max_epochs', help='select dataset for training.', default=100)
        parser.add_argument('--startup', help='select dataset for training.', default=1)
        parser.add_argument('--dataset_dirs_train', help='select dataset for training.', default=None)
        parser.add_argument('--dataset_dirs_val', help='select dataset for training.', default=None)
        parser.add_argument('--dataset_dirs_test', help='select dataset for training.', default=None)
        parser.add_argument('--results_folder', help='select folder for results of inference.', default=None)
        parser.add_argument('--traintestmode', help='Train or test mode?', default=None)
        parser.add_argument('--netinfo', help='Which network is trained/evaluated?', default=None)
        parser.add_argument('--scaled', help='Images scaled to the same size?', default=None)
        parser.add_argument('--overlap_train', help='Images scaled to the same size?', default=None)
        parser.add_argument('--overlap_test', help='Images scaled to the same size?', default=None)

        args = parser.parse_args()
    except:
        e=1
    if (int(args.startup) == 0):
        print("Writing config...")
        os.remove("D:\DeepLearning\Kaggle\Config\config_spots.json")
        settings.writeValues(net_description=args.net_description,max_epochs=int(args.max_epochs),dataset=args.dataset,dataset_dirs_train=args.dataset_dirs_train,dataset_dirs_val=args.dataset_dirs_val,results_folder=args.results_folder,traintestmode=args.traintestmode,netinfo=args.netinfo,dataset_dirs_test=args.dataset_dirs_test,scaled=args.scaled,overlap_train=args.overlap_train,overlap_test=args.overlap_test)
    else:
        print("Startup mode")
        print(args.startup)
main()