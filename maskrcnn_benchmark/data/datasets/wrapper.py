import torch
from .abstract import AbstractDataset


class WrapperDataset(AbstractDataset):
    """
    When training on multiple datasets class labels can be misaligned, and our
    last hope is to look at human readable class names to find a mapping

    This auxiliary dataset helps to find common class names between
    `mimicked_dataset` and `wrapped_dataset` and wrap the latter to return
    ground truth that aligns with the indices of the former.

        A: mimicked_dataset
        B: wrapped_dataset

    IMPORTANT:
    By design this wrapper utilizes fields and methods of datasets
    derived from AbstractDataset
    """

    def __init__(self, A, B):
        # A: mimicked_dataset
        # B: wrapped_dataset
        self.A = A
        self.B = B

        common_classes = set(A.CLASSES) & set(B.CLASSES)
        self.common_classes = common_classes
        assert len(common_classes) > 0

        self.idA_to_idB = {
            id: B.name_to_id[name] if name in B.name_to_id else None
            for id, name in A.id_to_name.items()
        }
        self.idB_to_idA = {
            id: A.name_to_id[name] if name in A.name_to_id else None
            for id, name in B.id_to_name.items()
        }

        # NOTE: By default ids go from 0 to N-1 to address all heads in the
        # RCNN RoI heads (contiguous id), and here we assume that the network
        # uses the `mimicked_dataset`'s classes, and all ids of the wrapper
        # will represent the corresponding class in the `mimicked_dataset`'s
        # indexing. Therefore by looking only at the wrapper's used IDs they may
        # not appear contiguous, still they are part of a contiguous mapping.

        # Resolving contiguous mapping by filling empty spots
        self.CLASSES = [
            name if name in common_classes else f"__unmatched__({name})"
            for name in A.CLASSES
        ]
        assert self.CLASSES[0] == "__background__"

        self.name_to_id = {name: id for id, name in enumerate(self.CLASSES)}
        self.id_to_name = {id: name for name, id in self.name_to_id.items()}

    def __getitem__(self, idx):
        img, target, idx = self.B[idx]
        labels = target.get_field("labels")

        # Remove objects from wrapped GT belonging to classes not present in
        # the mimicked dataset
        select_idx = torch.tensor(
            [self.idB_to_idA[idB] is not None for idB in labels.tolist()],
            dtype=torch.uint8,
        )

        # Fancy indexing using a boolean selection tensor
        labels = labels[select_idx]
        target.bbox = target.bbox[select_idx]

        if "masks" in target.fields():
            masks = target.get_field("masks")[select_idx]
            target.add_field("masks", masks)

        # Convert ids from wrapped to mimicked
        for i in range(len(labels)):
            labels[i] = self.idB_to_idA[labels[i].item()]
        target.add_field("labels", labels)
        return img, target, idx

    def get_img_info(self, idx):
        return self.B.get_img_info(idx)

    def __len__(self):
        return len(self.B)

    def __str__(self):
        r = (
            f"[WrapperDataset mimicks:{self.A.__class__.__name__} "
            f"wraps:{self.B.__class__.__name__}]"
            f"\n{'Mimicked index':>15} : {'Mimicked label':<15}    {'Wrapped index':>15} : {'Wrapped label':<15}\n"
        )
        for id, name in self.A.id_to_name.items():
            r += f"{id:>15} : {name:<15} -> {str(self.idA_to_idB[id]):>15} : {self.id_to_name[id]:<15}\n"
        return r
