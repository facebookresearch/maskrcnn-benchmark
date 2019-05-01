from __future__ import absolute_import, division, print_function, unicode_literals


def add_archs(archs):
    global MODEL_ARCH
    for x in archs:
        assert x not in MODEL_ARCH, "Duplicated model name {} existed".format(x)
        MODEL_ARCH[x] = archs[x]


MODEL_ARCH = {
    "default": {
        "block_op_type": [
            # stage 0
            ["ir_k3"],
            # stage 1
            ["ir_k3"] * 2,
            # stage 2
            ["ir_k3"] * 3,
            # stage 3
            ["ir_k3"] * 7,
            # stage 4, bbox head
            ["ir_k3"] * 4,
            # stage 5, rpn
            ["ir_k3"] * 3,
            # stage 5, mask head
            ["ir_k3"] * 5,
        ],
        "block_cfg": {
            "first": [32, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[6, 24, 2, 2]],
                # stage 2
                [[6, 32, 3, 2]],
                # stage 3
                [[6, 64, 4, 2], [6, 96, 3, 1]],
                # stage 4, bbox head
                [[4, 160, 1, 2], [6, 160, 2, 1], [6, 240, 1, 1]],
                # [[6, 160, 3, 2], [6, 320, 1, 1]],
                # stage 5, rpn head
                [[6, 96, 3, 1]],
                # stage 6, mask head
                [[4, 160, 1, 1], [6, 160, 3, 1], [3, 80, 1, -2]],
            ],
            # [c, channel_scale]
            "last": [0, 0.0],
            "backbone": [0, 1, 2, 3],
            "rpn": [5],
            "bbox": [4],
            "mask": [6],
        },
    },
    "xirb16d_dsmask": {
        "block_op_type": [
            # stage 0
            ["ir_k3"],
            # stage 1
            ["ir_k3"] * 2,
            # stage 2
            ["ir_k3"] * 3,
            # stage 3
            ["ir_k3"] * 7,
            # stage 4, bbox head
            ["ir_k3"] * 4,
            # stage 5, mask head
            ["ir_k3"] * 5,
            # stage 6, rpn
            ["ir_k3"] * 3,
        ],
        "block_cfg": {
            "first": [16, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[6, 32, 2, 2]],
                # stage 2
                [[6, 48, 3, 2]],
                # stage 3
                [[6, 96, 4, 2], [6, 128, 3, 1]],
                # stage 4, bbox head
                [[4, 128, 1, 2], [6, 128, 2, 1], [6, 160, 1, 1]],
                # stage 5, mask head
                [[4, 128, 1, 2], [6, 128, 2, 1], [6, 128, 1, -2], [3, 64, 1, -2]],
                # stage 6, rpn head
                [[6, 128, 3, 1]],
            ],
            # [c, channel_scale]
            "last": [0, 0.0],
            "backbone": [0, 1, 2, 3],
            "rpn": [6],
            "bbox": [4],
            "mask": [5],
        },
    },
    "mobilenet_v2": {
        "block_op_type": [
            # stage 0
            ["ir_k3"],
            # stage 1
            ["ir_k3"] * 2,
            # stage 2
            ["ir_k3"] * 3,
            # stage 3
            ["ir_k3"] * 7,
            # stage 4
            ["ir_k3"] * 4,
        ],
        "block_cfg": {
            "first": [32, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 16, 1, 1]],
                # stage 1
                [[6, 24, 2, 2]],
                # stage 2
                [[6, 32, 3, 2]],
                # stage 3
                [[6, 64, 4, 2], [6, 96, 3, 1]],
                # stage 4
                [[6, 160, 3, 1], [6, 320, 1, 1]],
            ],
            # [c, channel_scale]
            "last": [0, 0.0],
            "backbone": [0, 1, 2, 3],
            "bbox": [4],
        },
    },
}


MODEL_ARCH_CHAM = {
    "cham_v1a": {
        "block_op_type": [
            # stage 0
            ["ir_k3"],
            # stage 1
            ["ir_k7"] * 2,
            # stage 2
            ["ir_k3"] * 5,
            # stage 3
            ["ir_k5"] * 7 + ["ir_k3"] * 5,
            # stage 4, bbox head
            ["ir_k3"] * 5,
            # stage 5, rpn
            ["ir_k3"] * 3,
        ],
        "block_cfg": {
            "first": [32, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 24, 1, 1]],
                # stage 1
                [[4, 48, 2, 2]],
                # stage 2
                [[7, 64, 5, 2]],
                # stage 3
                [[12, 56, 7, 2], [8, 88, 5, 1]],
                # stage 4, bbox head
                [[7, 152, 4, 2], [10, 104, 1, 1]],
                # stage 5, rpn head
                [[8, 88, 3, 1]],
            ],
            # [c, channel_scale]
            "last": [0, 0.0],
            "backbone": [0, 1, 2, 3],
            "rpn": [5],
            "bbox": [4],
        },
    },
    "cham_v2": {
        "block_op_type": [
            # stage 0
            ["ir_k3"],
            # stage 1
            ["ir_k5"] * 4,
            # stage 2
            ["ir_k7"] * 6,
            # stage 3
            ["ir_k5"] * 3 + ["ir_k3"] * 6,
            # stage 4, bbox head
            ["ir_k3"] * 7,
            # stage 5, rpn
            ["ir_k3"] * 1,
        ],
        "block_cfg": {
            "first": [32, 2],
            "stages": [
                # [t, c, n, s]
                # stage 0
                [[1, 24, 1, 1]],
                # stage 1
                [[8, 32, 4, 2]],
                # stage 2
                [[5, 48, 6, 2]],
                # stage 3
                [[9, 56, 3, 2], [6, 56, 6, 1]],
                # stage 4, bbox head
                [[2, 160, 6, 2], [6, 112, 1, 1]],
                # stage 5, rpn head
                [[6, 56, 1, 1]],
            ],
            # [c, channel_scale]
            "last": [0, 0.0],
            "backbone": [0, 1, 2, 3],
            "rpn": [5],
            "bbox": [4],
        },
    },
}
add_archs(MODEL_ARCH_CHAM)
