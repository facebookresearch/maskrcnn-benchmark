import time
from datetime import datetime


def get_tensorboard_writer(local_rank, distributed):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        raise ImportError(
            'To use tensorboard please install tensorboardX '
            '[ pip install tensorflow tensorboardX ].'
        )

    if not distributed or (distributed and local_rank == 0):
        timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M')
        tb_logger = SummaryWriter('logs/maskrcnn-{}'.format(timestamp))
        return tb_logger
    else:
        return None
