import torch
# from torch.distributed import deprecated as dist

from maskrcnn_benchmark.config import cfg

from maskrcnn_benchmark.data.datasets import coco
from maskrcnn_benchmark.data.collate_batch import BatchCollator
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.data.build import make_data_sampler, make_batch_data_sampler



root = "./datasets/coco/val2014"
ann_file = "./datasets/coco/annotations/instances_debug2014.json" 
remove_images_without_annotations = True

shuffle = True
is_distributed = False # gpus > 1
images_per_batch = 2
num_gpus = 1
start_iter = 0
num_iters = 10
aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
num_workers = 2 #cfg.DATALOADER.NUM_WORKERS
images_per_gpu = images_per_batch // num_gpus
is_train = 1

transforms = build_transforms(cfg, is_train)

dataset = coco.COCODataset(ann_file, root, remove_images_without_annotations, transforms)

sampler = make_data_sampler(dataset, shuffle, is_distributed)
batch_sampler = make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter)
collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
data_loader = torch.utils.data.DataLoader(
    dataset,
    num_workers=num_workers,
    batch_sampler=batch_sampler,
    collate_fn=collator,
)

# MODEL
# model = build_detection_model(cfg)
# device = torch.device(cfg.MODEL.DEVICE)
# model.to(device)

for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
	t1 = targets[0]
	m1 = t1.get_field("masks")
	polygons = [x for x in m1]
	p1 = polygons[0]
	proposal = [0,0,100,100]
	cropped_p1 = p1.crop(proposal)
	resized_p1 = cropped_p1.resize((50,50))
	mask = resized_p1.convert('mask')
	mask_np = mask.numpy()
    # images = images.to(device)
    # targets = [target.to(device) for target in targets]


