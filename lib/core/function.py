import torch
import torch.nn as nn
import pandas as pd
import os

from core.loss import loss_function_ab, loss_function_af
from core.utils_box import reg2loc
from core.ab_match import anchor_box_adjust, anchor_bboxes_encode
from core.utils_ab import result_process_ab, result_process_af


dtype = torch.cuda.FloatTensor() if torch.cuda.is_available() else torch.FloatTensor()
dtypel = torch.cuda.LongTensor() if torch.cuda.is_available() else torch.LongTensor()


def ab_prediction_train(cfg, out_ab, label, boxes, action_num):
    '''
    Loss for anchor-based module includes: category classification loss, overlap loss and regression loss
    '''
    match_xs_ls = list()
    match_ws_ls = list()
    match_labels_ls = list()
    match_scores_ls = list()
    anchors_class_ls = list()
    anchors_x_ls = list()
    anchors_w_ls = list()
    anchors_rx_ls = list()
    anchors_rw_ls = list()

    for layer in range(cfg.MODEL.NUM_LAYERS):
        # anchors_class: bs, ti*n_box, nclass. others: bs, ti*n_box
        match_xs, match_ws, match_scores, match_labels, \
        anchors_x, anchors_w, anchors_rx, anchors_rw, anchors_class = \
            anchor_bboxes_encode(cfg, out_ab[layer], label, boxes, action_num, layer)

        match_xs_ls.append(match_xs)
        match_ws_ls.append(match_ws)
        match_scores_ls.append(match_scores)
        match_labels_ls.append(match_labels)

        anchors_x_ls.append(anchors_x)
        anchors_w_ls.append(anchors_w)
        anchors_rx_ls.append(anchors_rx)
        anchors_rw_ls.append(anchors_rw)
        anchors_class_ls.append(anchors_class)

    # collect the predictions
    match_xs_ls = torch.cat(match_xs_ls, dim=1)
    match_ws_ls = torch.cat(match_ws_ls, dim=1)
    match_labels_ls = torch.cat(match_labels_ls, dim=1)
    match_scores_ls = torch.cat(match_scores_ls, dim=1)
    anchors_class_ls = torch.cat(anchors_class_ls, dim=1)
    anchors_x_ls = torch.cat(anchors_x_ls, dim=1)
    anchors_w_ls = torch.cat(anchors_w_ls, dim=1)
    anchors_rx_ls = torch.cat(anchors_rx_ls, dim=1)
    anchors_rw_ls = torch.cat(anchors_rw_ls, dim=1)

    return anchors_x_ls, anchors_w_ls, anchors_rx_ls, anchors_rw_ls, anchors_class_ls, \
           match_xs_ls, match_ws_ls, match_scores_ls, match_labels_ls


def ab_predict_eval(cfg, out_ab):
    # collect predictions
    anchors_class_ls = list()
    anchors_x_ls = list()
    anchors_w_ls = list()

    for layer in range(cfg.MODEL.NUM_LAYERS):
        anchors_class, anchors_x, anchors_w = anchor_box_adjust(cfg, out_ab[layer], layer)
        anchors_class_ls.append(anchors_class)
        anchors_x_ls.append(anchors_x)
        anchors_w_ls.append(anchors_w)

    # classification score
    anchors_class_ls = torch.cat(anchors_class_ls, dim=1)
    # regression
    anchors_x_ls = torch.cat(anchors_x_ls, dim=1)
    anchors_w_ls = torch.cat(anchors_w_ls, dim=1)

    return anchors_class_ls, anchors_x_ls, anchors_w_ls


def train(cfg, train_loader, model, optimizer):
    model.train()
    loss_record = 0
    cls_loss_ab_record = 0
    loc_loss_ab_record = 0

    for feat_spa, feat_tem, boxes, label, action_num in train_loader:
        optimizer.zero_grad()

        # feature = feat_spa
        feature = torch.cat((feat_spa, feat_tem), dim=1)
        feature = feature.type_as(dtype)
        boxes = boxes.float().type_as(dtype)
        label = label.type_as(dtypel)

        out_ab = model(feature)

        # Loss for anchor-based module, including clasification loss, overlap loss and regression loss
        # anchors_class_ls: bs, sum_i(ti*n_box), n_class. others: bs, sum_i(ti*n_box)
        anchors_x_ls, anchors_w_ls, anchors_rx_ls, anchors_rw_ls, anchors_class_ls, \
        match_xs_ls, match_ws_ls, match_scores_ls, match_labels_ls = ab_prediction_train(cfg, out_ab, label, boxes, action_num)
        cls_loss_ab, loc_loss_ab = loss_function_ab(anchors_x_ls, anchors_w_ls, anchors_rx_ls, anchors_rw_ls,
                                                    anchors_class_ls, match_xs_ls, match_ws_ls,
                                                    match_scores_ls, match_labels_ls, cfg)

        # complete loss
        loss = cls_loss_ab + loc_loss_ab

        loss.backward()
        optimizer.step()
        loss_record = loss_record + loss.item()

        cls_loss_ab_record = cls_loss_ab_record + cls_loss_ab.item()
        loc_loss_ab_record = loc_loss_ab_record + loc_loss_ab.item()

    loss_avg = loss_record / len(train_loader)

    cls_loss_ab_record_avg = cls_loss_ab_record / len(train_loader)
    loc_loss_ab_record_avg = loc_loss_ab_record / len(train_loader)

    return loss_avg, cls_loss_ab_record_avg, loc_loss_ab_record_avg


def evaluation(val_loader, model, epoch, cfg):
    model.eval()

    out_df_ab = pd.DataFrame(columns=cfg.TEST.OUTDF_COLUMNS_AB)
    for feat_spa, feat_tem, begin_frame, video_name in val_loader:
        begin_frame = begin_frame.detach().numpy()  # 0 for anet
        video_len = cfg.DATASET.WINDOW_SIZE if cfg.DATASET.NAME == 'THUMOS14' else 1.0

        # feature = feat_spa
        feature = torch.cat((feat_spa, feat_tem), dim=1)
        feature = feature.type_as(dtype)
        out_ab = model(feature)

        ############################### anchor-based ###############################
        # collect predictions
        anchors_class_ls, anchors_x_ls, anchors_w_ls = ab_predict_eval(cfg, out_ab)

        # classification score
        # anchors_class_ls = torch.sigmoid(anchors_class_ls)
        m = nn.Softmax(dim=2).cuda()
        anchors_class_ls = m(anchors_class_ls)
        cls_score = anchors_class_ls.detach().cpu().numpy()

        # regression
        anchors_xmins = anchors_x_ls - anchors_w_ls / 2
        tmp_xmins = anchors_xmins.detach().cpu().numpy()
        xmins = tmp_xmins * video_len

        anchors_xmaxs = anchors_x_ls + anchors_w_ls / 2
        tmp_xmaxs = anchors_xmaxs.detach().cpu().numpy()
        xmaxs = tmp_xmaxs * video_len

        tmp_df_ab = result_process_ab(video_name, video_len, begin_frame, cls_score, xmins, xmaxs, cfg)

        out_df_ab = pd.concat([out_df_ab, tmp_df_ab])
        ############################### anchor-based ###############################

    if cfg.BASIC.SAVE_PREDICT_RESULT:
        predict_file = os.path.join(cfg.BASIC.ROOT_DIR, cfg.TEST.PREDICT_CSV_FILE+'_ab'+str(epoch)+'.csv')
        print('predict_file', predict_file)
        out_df_ab.to_csv(predict_file, index=False)

    return out_df_ab
