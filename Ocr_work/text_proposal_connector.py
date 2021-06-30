import numpy as np

def threshold(coords, min_, max_):
    return np.maximum(np.minimum(coords, max_), min_)
    
def clip_boxes(boxes, im_shape):
    boxes[:, 0::2] = threshold(boxes[:, 0::2], 0, im_shape[1])
    boxes[:, 1::2] = threshold(boxes[:, 1::2], 0, im_shape[0])
    return boxes

class Graph(object):
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        # subgraph里面的每一个列表都是一个文本框
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            # self.graph[i,j]=True代表第i个框接上了第j个框
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs


class TextProposalGraphBuilder(object):
    def __init__(self, max_horizontal_gap=50, min_vertical_overlaps=0.7, min_size_similarity=0.7):
        #------------------------------------------------#
        #    max_horizontal_gap: 文本行内，文本框最大水平距离,超出此距离的文本框属于不同的文本行
        #    min_vertical_overlaps：文本框最小垂直Iou
        #    min_size_similarity: 文本框尺寸最小相似度
        #------------------------------------------------#

        self.max_horizontal_gap = max_horizontal_gap
        self.min_vertical_overlaps = min_vertical_overlaps
        self.min_size_similarity = min_size_similarity
        self.text_proposals = None
        self.scores = None
        self.im_size = None
        self.heights = None
        self.boxes_table = None

    def meet_v_iou(self, index1, index2):
        #---------------------------------#
        #   判断两个文本框是否满足垂直条件
        #---------------------------------#
        def overlaps_v(idx1, idx2):
            # 高的重合程度
            h1 = self.heights[idx1]
            h2 = self.heights[idx2]
            # 垂直方向的交集
            max_y1 = max(self.text_proposals[idx2][0], self.text_proposals[idx1][0])
            min_y2 = min(self.text_proposals[idx2][2], self.text_proposals[idx1][2])
            return max(0, min_y2 - max_y1) / min(h1, h2)

        def size_similarity(idx1, idx2):
            # 高的相似程度
            h1 = self.heights[idx1]
            h2 = self.heights[idx2]
            return min(h1, h2) / max(h1, h2)

        return overlaps_v(index1, index2) >= self.min_vertical_overlaps and \
               size_similarity(index1, index2) >= self.min_size_similarity

    def get_successions(self, index):
        #------------------------------------------------#
        #   获取指定索引号文本框的后继文本框
        #   index: 文本框索引号
        #   所有后继文本框的索引号列表
        #------------------------------------------------#
        box = self.text_proposals[index]
        results = []
        # 判断当前框是否由相邻的后继的框
        for left in range(int(box[1]) + 1, min(int(box[1]) + self.max_horizontal_gap + 1, self.im_size[1])):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                # 判断高的重合情况
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        #-----------------------------------#
        #   获取指定索引号文本框的前驱文本框
        #   index: 文本框索引号
        #   获得所有前驱文本框的索引号列表
        #-----------------------------------#  
        box = self.text_proposals[index]
        results = []
        # 向前遍历
        for left in range(int(box[1]) - 1, max(int(box[1] - self.max_horizontal_gap), 0) - 1, -1):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def build_graph(self, text_proposals, scores, im_size):
        #---------------------------------------------------------------#
        #   根据文本框构建文本框对
        #   text_proposals: 文本框，numpy 数组，[n,(y1,x1,y2,x2)]
        #   scores: 文本框得分，[n]
        #   im_size: 图像尺寸,tuple(H,W,C)
        #   返回二维bool类型 numpy数组，[n,n]；指示文本框两两之间是否配对
        #---------------------------------------------------------------#
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        # 求取所有文本框的高度
        self.heights = text_proposals[:, 3] - text_proposals[:, 1]
        
        # 安装每个文本框左侧坐标x1分组
        im_width = self.im_size[1]
        self.boxes_table = [[] for _ in range(im_width)]
        # 根据每一列进行划分
        for index, box in enumerate(text_proposals):
            self.boxes_table[int(box[1])].append(index)

        # 用于表示几个文本框之间的连接情况
        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            # 获取当前文本框(Bi)的后继文本框
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            # 后继文本框中得分最高的那个，记做Bj
            succession_index = successions[np.argmax(scores[successions])]
            # 获取Bj的前驱文本框
            precursors = self.get_precursors(succession_index)
            # 根据得分判断是否构成前后结点
            if self.scores[index] >= np.max(self.scores[precursors]):
                graph[index, succession_index] = True
        return Graph(graph)


class TextProposalConnector:
    # 连接文本框，组成文本行
    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def fit_y(self, X, Y, x1, x2):
        #--------------------------------------------#
        #   一元线性函数拟合X,Y,并返回x1,x2的的函数值
        #--------------------------------------------#
        len(X) != 0
        # 只有一个点返回 y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        #------------------------------------------#
        #   获取文本行
        #   text_proposals: 文本框，[n,(y1,x1,y2,x2)]
        #   scores: 文本框得分，[n]
        #   im_size: 图像尺寸,tuple(H,W,C)
        #   return: 文本行，边框和得分,numpy数组
        #------------------------------------------#
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        tp_groups = graph.sub_graphs_connected()

        # text_lines里面每一个列表代表一个文本行，每一个列表保存该文本行对应的所有文本框
        text_lines = np.zeros((len(tp_groups), 9), np.float32)
        # 逐个文本行处理
        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]
            # 宽度方向最小值和最大值
            x_min = np.min(text_line_boxes[:, 1])
            x_max = np.max(text_line_boxes[:, 3])
            # 文本框宽度的一半
            offset = (text_line_boxes[0, 3] - text_line_boxes[0, 1]) * 0.5
            # 使用一元线性函数求文本行左右两边高度边界
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 1], text_line_boxes[:, 0], x_min - offset, x_max + offset)
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 1], text_line_boxes[:, 2], x_min - offset, x_max + offset)

            # 文本行的得分为所有文本框得分的均值
            score = scores[list(tp_indices)].sum() / float(len(tp_indices))
            # 文本行坐标
            text_lines[index, 0] = x_min
            text_lines[index, 1] = lt_y
            text_lines[index, 2] = x_max
            text_lines[index, 3] = rt_y
            text_lines[index, 4] = x_max
            text_lines[index, 5] = rb_y
            text_lines[index, 6] = x_min
            text_lines[index, 7] = lb_y
            text_lines[index, 8] = score
        # 裁剪到图像尺寸内
        text_lines = clip_boxes(text_lines, im_size)

        return text_lines
