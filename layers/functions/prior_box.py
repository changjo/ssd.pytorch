import torch
from math import sqrt as sqrt
from itertools import product as product


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    # PriorBox의 cfg..
    # 값들이 hard coding 되어 있다..
    # hard coding된 값들을 이용하여 역으로 scale 등을 계산한다..
    # v2 = {
    #     'feature_maps' : [38, 19, 10, 5, 3, 1],
    #     'min_dim' : 300,
    #     'steps' : [8, 16, 32, 64, 100, 300],
    ##
    ##    'min_sizes' : 논문에 있는 scale 계산법에 이미지 크기 300을 곱한 결과를 미리 계산
    ##                  한 것 같다.
    #     'min_sizes' : [30, 60, 111, 162, 213, 264],
    ##
    ##    'max_sizes' : 논문에 aspect ratio가 1인 경우에 scale을 계산하는 방법이 나와
    ##                  있는데 "s_k_prime = sqrt(s_k * s_k+1)"
    ##                  s_k_prime을 구하기 위해 필요한 s_k+1 곲하기 300을 미리 계산했다.
    #     'max_sizes' : [60, 111, 162, 213, 264, 315],
    #
    #     # 'aspect_ratios' : [[2, 1/2], [2, 1/2, 3, 1/3], [2, 1/2, 3, 1/3],
    #     #                    [2, 1/2, 3, 1/3], [2, 1/2], [2, 1/2]],
    #     'aspect_ratios' : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    #     'variance' : [0.1, 0.2],
    #     'clip' : True,
    #     'name' : 'v2',
    # }

    def forward(self):
        mean = []
        # TODO merge these
        if self.version == 'v2':
            for k, f in enumerate(self.feature_maps):
                for i, j in product(range(f), repeat=2):
                    # f_k = 300 / self.steps[k]
                    # 왜 이렇게 했을까? f_k는 feature_maps에 이미 정의 되어있는것 같은데..
                    f_k = self.image_size / self.steps[k]
                    # unit center x,y
                    # 좌표들을 (0, 1) 범위로 normalize 한다.
                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k

                    # aspect_ratio: 1 일 때,
                    # rel size: min_size
                    # scale 계산.. 왜 논문에 있는데로 안 했을까?..
                    s_k = self.min_sizes[k]/self.image_size
                    mean += [cx, cy, s_k, s_k]

                    # aspect_ratio: 1 일 때,
                    # rel size: sqrt(s_k * s_(k+1))
                    # s_(k+1) = self.max_sizes[k]/self.image_size
                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))

                    # s_k 또는 s_k_prime은 normalize 된 width 또는 height로 생각할
                    # 수 있다.
                    mean += [cx, cy, s_k_prime, s_k_prime]

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:

                        # 논문에 나와 있는데로, w=s_k*sqrt(ar), h=s_k/sqrt(ar)
                        mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                        mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        else:
            # original version generation of prior (default) boxes
            for i, k in enumerate(self.feature_maps):
                step_x = step_y = self.image_size/k
                for h, w in product(range(k), repeat=2):
                    c_x = ((w+0.5) * step_x)
                    c_y = ((h+0.5) * step_y)
                    c_w = c_h = self.min_sizes[i] / 2
                    s_k = self.image_size  # 300
                    # aspect_ratio: 1,
                    # size: min_size
                    mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                             (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    if self.max_sizes[i] > 0:
                        # aspect_ratio: 1
                        # size: sqrt(min_size * max_size)/2
                        c_w = c_h = sqrt(self.min_sizes[i] *
                                         self.max_sizes[i])/2
                        mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                 (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    # rest of prior boxes
                    for ar in self.aspect_ratios[i]:
                        if not (abs(ar-1) < 1e-6):
                            c_w = self.min_sizes[i] * sqrt(ar)/2
                            c_h = self.min_sizes[i] / sqrt(ar)/2
                            mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                     (c_x+c_w)/s_k, (c_y+c_h)/s_k]
        # back to torch land
        ## 리스트 mean을 n x 4 torch tensor로 만든다.
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            ## 값이 max보다 크면 1로, min보다 작으면 0으로 값을 바꾼다.
            output.clamp_(max=1, min=0)
        return output
