import torch.nn as nn
import torch


class BaseFeatureNet(nn.Module):
    '''
    calculate feature
    input: [batch_size, 128, 1024]
    output: [batch_size, 32, 512]
    '''
    def __init__(self, cfg):
        super(BaseFeatureNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=cfg.MODEL.IN_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.max_pooling1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=cfg.MODEL.BASE_FEAT_DIM,
                               out_channels=cfg.MODEL.BASE_FEAT_DIM,
                               kernel_size=9, stride=1, padding=4, bias=True)
        self.max_pooling2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.relu(self.conv1(x))
        feat = self.max_pooling1(feat)
        feat = self.relu(self.conv2(feat))
        out = self.max_pooling2(feat)
        return out


class MainAnchorNet(nn.Module):
    '''
    main network
    input: base feature, [batch_size, 32, 512]
    output: MAL1, MAL2, MAL3
    '''
    def __init__(self, cfg):
        super(MainAnchorNet, self).__init__()
        self.convs = nn.ModuleList()
        for layer in range(cfg.MODEL.NUM_LAYERS):
            in_channel = cfg.MODEL.BASE_FEAT_DIM if layer == 0 else cfg.MODEL.LAYER_DIMS[layer-1]
            out_channel = cfg.MODEL.LAYER_DIMS[layer]
            conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=cfg.MODEL.LAYER_STRIDES[layer], padding=1)
            self.convs.append(conv)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        results = []
        feat = x
        for conv in self.convs:
            feat = self.relu(conv(feat))
            results.append(feat)

        return tuple(results)


class ReduceChannel(nn.Module):
    def __init__(self, cfg):
        super(ReduceChannel, self).__init__()
        self.convs = nn.ModuleList()
        for layer in range(cfg.MODEL.NUM_LAYERS):
            conv = nn.Conv1d(cfg.MODEL.LAYER_DIMS[layer], cfg.MODEL.REDU_CHA_DIM, kernel_size=1)
            self.convs.append(conv)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat_list):
        assert len(feat_list) == len(self.convs)
        results = []
        for conv, feat in zip(self.convs, feat_list):
            results.append(self.relu(conv(feat)))

        return tuple(results)


class SSAD(nn.Module):
    '''
    Action multi-class classification and regression
    input:
    output: class score + conf + location (center & width)
    '''
    def __init__(self, cfg):
        super(SSAD, self).__init__()
        self.num_class = cfg.DATASET.NUM_CLASSES
        self.num_pred_value = cfg.DATASET.NUM_CLASSES + 2
        self.base_feature_net = BaseFeatureNet(cfg)
        self.main_anchor_net = MainAnchorNet(cfg)
        # self.reduce_channel = ReduceChannel(cfg)

        self.num_box = len(cfg.MODEL.ASPECT_RATIOS)
        self.head_stack_layers = cfg.MODEL.HEAD_LAYERS
        self._init_head(cfg)

    def _init_head(self, cfg):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for layer in range(self.head_stack_layers):
            in_channel = cfg.MODEL.REDU_CHA_DIM if layer == 0 else cfg.MODEL.HEAD_DIM
            out_channel = cfg.MODEL.HEAD_DIM
            cls_conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1)
            reg_conv = nn.Conv1d(in_channel, out_channel, kernel_size=3, padding=1)
            self.cls_convs.append(cls_conv)
            self.reg_convs.append(reg_conv)
        self.pred_cls = nn.Conv1d(cfg.MODEL.HEAD_DIM, self.num_box * self.num_class, kernel_size=3, padding=1)
        self.pred_reg = nn.Conv1d(cfg.MODEL.HEAD_DIM, self.num_box * 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def tensor_view(self, cls, reg):
        '''
        view the tensor for [batch, 120, depth] to [batch, (depth*5), 24]
        make the prediction (24 values) for each anchor box at the last dimension
        '''
        bs, c, t = cls.size()
        cls = cls.view(bs, -1, self.num_class, t).permute(0, 3, 1, 2).contiguous()
        reg = reg.view(bs, -1, 2, t).permute(0, 3, 1, 2).contiguous()
        data = torch.cat((cls, reg), dim=-1)
        data = data.view(bs, -1, self.num_pred_value)
        return data

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = self.relu(cls_conv(cls_feat))
        for reg_conv in self.reg_convs:
            reg_feat = self.relu(reg_conv(reg_feat))
        pred_cls = self.pred_cls(cls_feat)
        pred_reg = self.pred_reg(reg_feat)
        return self.tensor_view(pred_cls, pred_reg)

    def forward(self, x):
        base_feature = self.base_feature_net(x)
        feats = self.main_anchor_net(base_feature)
        # feats = self.reduce_channel(feats)

        return tuple(map(self.forward_single, feats))


if __name__ == '__main__':
    import sys
    sys.path.append('/home/data5/cxl/SSAD_v1/lib')
    from config import cfg, update_config
    cfg_file = '/home/data5/cxl/SSAD_v1/experiments/thumos/SSAD_train.yaml'
    update_config(cfg_file)

    model = SSAD(cfg)
    data = torch.randn((2, 2048, 128))
    out = model(data)
    for i in out:
        print(i.size())
