import os
import json
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TALDataset(Dataset):
    def __init__(self, cfg, split):
        self.root = os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.FEAT_DIR)
        self.split = split
        self.train_split = cfg.DATASET.TRAIN_SPLIT
        self.max_segment_num = cfg.DATASET.MAX_SEGMENT_NUM
        self.num_classes = cfg.DATASET.NUM_CLASSES
        self.base_dir = os.path.join(self.root, self.split)
        self.datas = self._make_dataset()

        self.name2idx = pickle.load(open(os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.NAME_IDX_FILE), 'rb'))
        self.video_anno = json.load(open(os.path.join(cfg.BASIC.ROOT_DIR, cfg.DATASET.GT_FILE)))['database']

    def __len__(self):
        return len(self.datas)

    def get_anno(self, video_name):
        label = list()
        box = list()
        video_info = self.video_anno[video_name]
        assert video_info['subset'] == self.split
        video_duration = video_info['duration']
        annos = video_info['annotations']
        for anno in annos:
            label.append(self.name2idx[anno['label']])
            start, end = anno['segment']
            box.append([start/video_duration, end/video_duration])

        box = np.array(box).astype('float32')
        label = np.array(label)
        return label, box

    def __getitem__(self, idx):
        file_name = self.datas[idx]
        data = np.load(os.path.join(self.base_dir, file_name))

        feat_tem = data['feat_tem']
        # feat_tem = cv2.resize(feat_tem, self.target_size, interpolation=cv2.INTER_LINEAR)
        feat_spa = data['feat_spa']
        # feat_spa = cv2.resize(feat_spa, self.target_size, interpolation=cv2.INTER_LINEAR)

        # pass video_name vis list
        video_name = str(data['vid_name'])
        if self.split == self.train_split:
            # data for anchor-based
            label, action = self.get_anno(video_name)
            num_segment = action.shape[0]
            assert num_segment > 0, 'no action in {}!!!'.format(video_name)
            action_padding = np.zeros((self.max_segment_num, 2), dtype=np.float)
            action_padding[:num_segment, :] = action
            label_padding = np.zeros(self.max_segment_num, dtype=np.int)
            label_padding[:num_segment] = label

            return feat_spa, feat_tem, action_padding, label_padding, num_segment
        else:
            return feat_spa, feat_tem, 0, video_name

    def _make_dataset(self):
        datas = os.listdir(self.base_dir)
        datas = [i for i in datas if i.endswith('.npz')]
        return datas


if __name__ == '__main__':
    import sys
    sys.path.append('/home/data5/cxl/SSAD/lib')
    from config import cfg, update_config

    cfg_file = '/home/data5/cxl/SSAD/experiments/thumos/SSAD.yaml'
    update_config(cfg_file)
    train_dset = TALDataset(cfg, cfg.DATASET.TRAIN_SPLIT, mode='train')
    train_loader = DataLoader(train_dset, batch_size=2, shuffle=True)

    for feat_spa, feat_tem, action, label, num in train_loader:
        print(type(feat_spa), feat_spa.size(), feat_tem.size())
