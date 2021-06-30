import numpy as np
from PIL import Image
from .text_proposal_connector import TextProposalConnector
from shapely.geometry import Polygon

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    x_offset,y_offset = (w-nw)//2/300, (h-nh)//2/300
    return new_image,x_offset,y_offset

def CTPN_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes

class BBoxUtility(object):
    def __init__(self, config, anchors=None, overlap_threshold=0.7,ignore_threshold=0.5,
                 nms_thresh=0.3):
        self.anchors = anchors
        self.num_anchors = 0 if anchors is None else len(anchors)
        self.overlap_threshold = overlap_threshold
        self.ignore_threshold = ignore_threshold
        self._nms_thresh = nms_thresh
        self.config = config
        self.text_proposal_connector = TextProposalConnector()

    def iou(self, box):
        # 计算出每个真实框与所有的先验框的iou
        # 判断真实框与先验框的重合情况
        inter_upleft = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # 真实框的面积
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        # 先验框的面积
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0])*(self.anchors[:, 3] - self.anchors[:, 1])
        # 计算iou
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

    def encode_box(self, box, return_iou=True):
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 3 + return_iou))

        # 找到每一个真实框，重合程度较高的先验框
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        
        # 找到对应的先验框
        assigned_anchors = self.anchors[assign_mask]
        # 逆向编码，将真实框转化为FasterRCNN预测结果的格式
        # 先计算真实框的中心与长宽
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        box_center = 0.5 * (box[:2] + box[2:])
        # 再计算重合度较高的先验框的中心与长宽
        assigned_anchors_w = (assigned_anchors[:, 2] - assigned_anchors[:, 0])
        assigned_anchors_h = (assigned_anchors[:, 3] - assigned_anchors[:, 1])
        assigned_anchors_center = 0.5 * (assigned_anchors[:, :2] + assigned_anchors[:, 2:4])
        

        # 逆向求取FasterRCNN应该有的预测结果
        encoded_box[:, 0][assign_mask] = box_center[1] - assigned_anchors_center[:,1]
        encoded_box[:, 0][assign_mask] /= assigned_anchors_h

        encoded_box[:, 1][assign_mask] = np.log(box_h / assigned_anchors_h)

        encoded_box[:, 2][assign_mask] = (box_center[0] - assigned_anchors_center[:,0]) * 2
        encoded_box[:, 2][assign_mask] /= assigned_anchors_w

        encoded_box[:, :3][assign_mask] = encoded_box[:, :3][assign_mask] / np.array(self.config.VARIANCE)
        return encoded_box.ravel()

    def ignore_box(self, box):
        iou = self.iou(box)
        
        ignored_box = np.zeros((self.num_anchors, 1))

        # 找到每一个真实框，重合程度较高的先验框
        assign_mask = (iou > self.ignore_threshold)&(iou<self.overlap_threshold)

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
            
        ignored_box[:, 0][assign_mask] = iou[assign_mask]
        return ignored_box.ravel()


    def assign_boxes(self, boxes):
        assignment = np.zeros((self.num_anchors, 3 + 1 + 1))

        assignment[:, -2] = 1.0
        if len(boxes) == 0:
            return assignment
            
        # 对每一个真实框都进行iou计算
        ingored_boxes = np.apply_along_axis(self.ignore_box, 1, boxes[:, :4])
        # 取重合程度最大的先验框，并且获取这个先验框的index
        ingored_boxes = ingored_boxes.reshape(-1, self.num_anchors, 1)
        # (num_anchors)
        ignore_iou = ingored_boxes[:, :, 0].max(axis=0)
        # (num_anchors)
        ignore_iou_mask = ignore_iou > 0

        assignment[:, -1][ignore_iou_mask] = -1

        # (n, num_anchors, 4)
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # 每一个真实框的编码后的值，和iou
        # (n, num_anchors)
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 4)

        # 取重合程度最大的先验框，并且获取这个先验框的index
        # (num_anchors)
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        # (num_anchors)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        # (num_anchors)
        best_iou_mask = best_iou > 0
        # 某个先验框它属于哪个真实框
        best_iou_idx = best_iou_idx[best_iou_mask]

        assign_num = len(best_iou_idx)
        # 保留重合程度最大的先验框的应该有的预测结果
        # 哪些先验框存在真实框
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]

        assignment[:, :3][best_iou_mask] = encoded_boxes[best_iou_idx,np.arange(assign_num),:3]
        # 3代表为背景的概率，为0
        assignment[:, -2][best_iou_mask] = 0
        assignment[:, -1][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

    def decode_boxes(self, deltas, side_deltas, anchors, variances, use_side_refine=True):
        # 高度和宽度
        w = anchors[:, 2] - anchors[:, 0]
        h = anchors[:, 3] - anchors[:, 1]

        # 中心点坐标
        cx = (anchors[:, 2] + anchors[:, 0]) * 0.5
        cy = (anchors[:, 3] + anchors[:, 1]) * 0.5

        deltas = np.concatenate([deltas, side_deltas], axis=1)
        # 回归系数
        deltas *= variances
        dy, dh, dx = deltas[:, 0], deltas[:, 1], deltas[:, 2]

        # 中心坐标回归
        cy += dy * h
        # 侧边精调
        cx += dx * w
        # 高度和宽度回归
        h *= np.exp(dh)

        # 转为y1,x1,y2,x2
        y1 = cy - h * 0.5
        y2 = cy + h * 0.5
        x1 = np.maximum(cx - w * 0.5, 0.)  # 限制在窗口内,修复后继节点找不到对应的前驱节点
        x2 = cx + w * 0.5

        if use_side_refine:
            return np.stack([x1, y1, x2, y2], axis=1)
        else:
            return np.stack([anchors[:, 0], y1, anchors[:, 2], y2], axis=1)


    def detection_out(self, predictions, background_label_id=0, 
                      confidence_threshold=0.5, variances = [0,1, 0.2, 0.1]):
        # 网络预测的结果
        class_scores, predict_deltas, predict_side_deltas = predictions[0][0], predictions[1][0], predictions[2][0]
        
        decode_bbox = self.decode_boxes(predict_deltas, predict_side_deltas, self.anchors, variances)

        class_conf = class_scores[:,1:2]
        
        conf_mask = (class_conf >= confidence_threshold)[:,0]

        detection = np.concatenate((decode_bbox[conf_mask], class_conf[conf_mask]), 1)

        best_box = []
        scores = detection[:,4]
        # 根据得分对该种类进行从大到小排序。
        arg_sort = np.argsort(scores)[::-1]
        detection = detection[arg_sort]
        while np.shape(detection)[0]>0:
            # 每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
            best_box.append(detection[0])
            if len(detection) == 1:
                break
            ious = iou(best_box[-1],detection[1:])
            detection = detection[1:][ious<self._nms_thresh]
        print(np.shape(best_box))
        return np.array(best_box)
    
    def combine(self, text_proposals, scores, image_shape):
        text_lines = self.text_proposal_connector.get_text_lines(text_proposals, scores, image_shape)
        keep_indices = self.filter_boxes(text_lines)
        text_lines = text_lines[keep_indices]

        # 文本行nms
        if text_lines.shape[0] != 0:
            keep_indices = quadrangle_nms(text_lines[:, :8], text_lines[:, 8],
                                                   self.config.TEXT_LINE_NMS_THRESH)
            text_lines = text_lines[keep_indices]

        return text_lines

    def filter_boxes(self, text_lines):
        widths = text_lines[:, 2] - text_lines[:, 0]
        scores = text_lines[:, -1]
        return np.where((scores > self.config.LINE_MIN_SCORE) &
                        (widths > (self.config.TEXT_PROPOSALS_WIDTH * self.config.MIN_NUM_PROPOSALS)))[0]



def quadrangle_iou(quadrangle_a, quadrangle_b):
    a = Polygon(quadrangle_a.reshape((4, 2)))
    b = Polygon(quadrangle_b.reshape((4, 2)))
    if not a.is_valid or not b.is_valid:
        return 0
    inter = Polygon(a).intersection(Polygon(b)).area
    union = a.area + b.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


def quadrangle_nms(quadrangles, scores, iou_threshold):
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        # 选择得分最高的
        i = order[0]
        keep.append(i)
        # 逐个计算iou
        overlap = np.array([quadrangle_iou(quadrangles[i], quadrangles[t]) for t in order[1:]])
        # 小于阈值的,用于下一个极值点选择
        indices = np.where(overlap < iou_threshold)[0]
        order = order[indices + 1]

    return keep

def iou(b1,b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    
    area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
    area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)
    
    iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
    return iou