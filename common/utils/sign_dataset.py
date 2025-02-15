import json
import os

from utils.dataset import Dataset


class SignDataset(Dataset):
    def _load_images_and_gt(self):
        gt_file = open(os.path.join(self.path, self.split + '.json'))
        gt_json = json.loads(gt_file.read())

        self.images = []
        self.gt = []

        for entry in gt_json:
            self.images.append(os.path.join(self.path, entry['file_name']))
            self.gt.append(entry['bounding_boxes'])
