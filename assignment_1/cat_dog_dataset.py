""" CODE PARTIALLY PROVIDED IN ASSIGNMENT. """

import os
import glob
import logging
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image


class CatDogDataset(Dataset):
    def __init__(
        self, 
        img_dir, 
        ann_dir, 
        input_img_size: int, 
        grid_size: int,
        logger: logging.Logger,
        transform=None
    ):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.input_img_size = input_img_size
        self.grid_size = grid_size
        self.logger = logger
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
        self.label_map = {"cat": 0, "dog": 1}  # Label mapping
        self.classes = list(self.label_map.keys())
        self._cache_targets()
        self.logger.debug(f"Initialised dataset, containing {len(self)} items")

    def parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        
        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            # Default to -1 if unknown label
            label = self.label_map.get(name, -1)  
            objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})

        return width, height, objects

    @staticmethod
    def xyxy_to_cxcywh_normalised(
        xmin: torch.Tensor | int | float, 
        ymin: torch.Tensor | int | float, 
        xmax: torch.Tensor | int | float, 
        ymax: torch.Tensor | int | float, 
        orig_w: torch.Tensor | int | float, 
        orig_h: torch.Tensor | int | float
    )-> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] |\
    tuple[float, float, float, float]:
        """
        Convert pixel-space [xmin, ymin, xmax, ymax] to
        image-normalized [cx, cy, w, h] in [0, 1].
        
        :param xmin: The x coord of top left corner.
        :type xmin: torch.Tensor | int | float
        :param ymin: The y coord of top left corner.
        :type ymin: torch.Tensor | int | float
        :param xmax: The x coord of bottom right corner.
        :type xmax: torch.Tensor | int | float
        :param ymax: The y coord of bottom right corner.
        :type ymax: torch.Tensor | int | float
        :param orig_w: Original width.
        :type orig_w: torch.Tensor | int | float
        :param orig_h: Original height. 
        :type orig_h: torch.Tensor | int | float
        :returns: Centre x, Centre y, width, height.
        :rtype: tuple[
                torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
            ] | tuple[float, float, float, float]
        """
        cx = ((xmin + xmax) / 2.0) / orig_w
        cy = ((ymin + ymax) / 2.0) / orig_h
        w  = (xmax - xmin) / orig_w
        h  = (ymax - ymin) / orig_h
        return cx, cy, w, h
    
    def build_yolo_target(self, norm_bboxes: list, labels: list):
        """
        Transform the cx, cy, width, height labelling to the cube
        expected by YOLO.

        :param norm_bboxes: Bounding boxes, relative to image.
        :type norm_bboxes: list
        :param labels: corresponding labels
        :type labels: list
        :returns: YOLO compatible target.
        :rtype: torch.Tensor
        """
        target = torch.zeros(
            self.grid_size, 
            self.grid_size, 
            # 5 for [x, y, w, y, objectness]
            5 + len(self.label_map)
        )

        for (cx, cy, w, h), label in zip(norm_bboxes, labels):
            if label < 0:
                continue

            # Translate x, y to cell in grid.
            col = min(int(cx * self.grid_size), self.grid_size - 1)
            row = min(int(cy * self.grid_size), self.grid_size - 1)

            x_cell = cx * self.grid_size - col
            y_cell = cy * self.grid_size - row

            class_vec = torch.zeros(len(self.label_map))
            class_vec[label] = 1.0

            target[row, col, 0] = x_cell
            target[row, col, 1] = y_cell
            target[row, col, 2] = w
            target[row, col, 3] = h
            target[row, col, 4] = 1.0
            target[row, col, 5:] = class_vec

        return target
    
    def _cache_targets(self):
        """
        Converts all the x-y-x-y-label targets to the output format 
        expected by the YOLO models. Saves this in cache.
        """
        self.logger.debug("Starting caching targets")
        self._targets = []
        self._labels = []
        for i in range(len(self.ann_files)):
            ann_path = self.ann_files[i]
            width, height, objects = self.parse_annotation(ann_path)
            
            norm_bboxes, labels = [], []
            for obj in objects:
                xmin, ymin, xmax, ymax = obj["bbox"]
                cx, cy, w, h = self.xyxy_to_cxcywh_normalised(
                    xmin, 
                    ymin, 
                    xmax, 
                    ymax, 
                    width, 
                    height
                )
                norm_bboxes.append((cx, cy, w, h))
                labels.append(obj["label"])
            
            # reshape to grid_size * grid_size * 7 
            self._targets.append(self.build_yolo_target(norm_bboxes, labels))
            self._labels.append(labels[0] if labels else -1)
        self.logger.debug("Done caching targets")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, self._targets[idx]
