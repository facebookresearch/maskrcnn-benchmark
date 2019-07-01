




class ShapeDataset(object):   
  
  def __init__(self, num_examples, transforms=None):
    
    self.height = 128
    self.width = 128
    
    self.num_examples = num_examples
    self.transforms = transforms # IMPORTANT, DON'T MISS
    self.image_info = []
    self.logger = logging.getLogger(__name__)
    
    # Class Names: Note that the ids start from 1, not 0. This repo uses the index 0 for background
    self.class_names = {"square": 1, "circle": 2, "triangle": 3}
    
    # Add images
    # Generate random specifications of images (i.e. color and
    # list of shapes sizes and locations). This is more compact than
    # actual images. Images are generated on the fly in load_image().
    for i in range(num_examples):
        bg_color, shapes = self.random_image(self.height, self.width)
        self.image_info.append({ "path":None,
                       "width": self.width, "height": self.height,
                       "bg_color": bg_color, "shapes": shapes
                       })
    
    # Fills in the self.coco varibale for evaluation.
    self.get_gt()
    
    # Variables needed for coco mAP evaluation
    self.id_to_img_map = {}
    for i, _ in enumerate(self.image_info):
      self.id_to_img_map[i] = i

    self.contiguous_category_id_to_json_id = { 0:0 ,1:1, 2:2, 3:3 }
    

  def random_shape(self, height, width):
    """Generates specifications of a random shape that lies within
    the given height and width boundaries.
    Returns a tuple of three values:
    * The shape name (square, circle, ...)
    * Shape color: a tuple of 3 values, RGB.
    * Shape dimensions: A tuple of values that define the shape size
                        and location. Differs per shape type.
    """
    # Shape
    shape = random.choice(["square", "circle", "triangle"])
    # Color
    color = tuple([random.randint(0, 255) for _ in range(3)])
    # Center x, y
    buffer = 20
    y = random.randint(buffer, height - buffer - 1)
    x = random.randint(buffer, width - buffer - 1)
    # Size
    s = random.randint(buffer, height//4)
    return shape, color, (x, y, s)

  def random_image(self, height, width):
      """Creates random specifications of an image with multiple shapes.
      Returns the background color of the image and a list of shape
      specifications that can be used to draw the image.
      """
      # Pick random background color
      bg_color = np.array([random.randint(0, 255) for _ in range(3)])
      # Generate a few random shapes and record their
      # bounding boxes
      shapes = []
      boxes = []
      N = random.randint(1, 4)
      labels = {}
      for _ in range(N):
          shape, color, dims = self.random_shape(height, width)
          shapes.append((shape, color, dims))
          x, y, s = dims
          boxes.append([y-s, x-s, y+s, x+s])

      # Apply non-max suppression with 0.3 threshold to avoid
      # shapes covering each other
      keep_ixs = non_max_suppression(np.array(boxes), np.arange(N), 0.3)
      shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
      
      return bg_color, shapes
  
  
  def draw_shape(self, image, shape, dims, color):
      """Draws a shape from the given specs."""
      # Get the center x, y and the size s
      x, y, s = dims
      if shape == 'square':
          cv2.rectangle(image, (x-s, y-s), (x+s, y+s), color, -1)
      elif shape == "circle":
          cv2.circle(image, (x, y), s, color, -1)
      elif shape == "triangle":
          points = np.array([[(x, y-s),
                              (x-s/math.sin(math.radians(60)), y+s),
                              (x+s/math.sin(math.radians(60)), y+s),
                              ]], dtype=np.int32)
          cv2.fillPoly(image, points, color)
      return image, [ x-s, y-s, x+s, y+s]


  def load_mask(self, image_id):
    """
    Generates instance masks for shapes of the given image ID.
    """
    info = self.image_info[image_id]
    shapes = info['shapes']
    count = len(shapes)
    mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
    
    boxes = []
    
    for i, (shape, _, dims) in enumerate(info['shapes']):
        mask[:, :, i:i+1], box = self.draw_shape(mask[:, :, i:i+1].copy(),
                                            shape, dims, 1)
        boxes.append(box)
        
    # Handle occlusions
    occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    for i in range(count-2, -1, -1):
        mask[:, :, i] = mask[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
    # Map class names to class IDs.
    class_ids = np.array([self.class_names[s[0]] for s in shapes])
    return mask.astype(np.uint8), class_ids.astype(np.int32), boxes
  
  def load_image(self, image_id):
    """Generate an image from the specs of the given image ID.
    Typically this function loads the image from a file, but
    in this case it generates the image on the fly from the
    specs in image_info.
    """
    # info = self.image_info[image_id]
    # bg_color = np.array(info['bg_color']).reshape([1, 1, 3])
    # image = np.ones([info['height'], info['width'], 3], dtype=np.uint8)
    # image = image * bg_color.astype(np.uint8)
    # for shape, color, dims in info['shapes']:
    #     image, _ = self.draw_shape(image, shape, dims, color)
    DATA_DIR = "/content/"
    folder = "maskrcnn-benchmark/data"
    image = cv2.imread(DATA_DIR + '/%s/%s.jpg'%(folder,image_id), cv2.IMREAD_COLOR)

    return image
      
  def __getitem__(self, idx):
    
    """Generate an image from the specs of the given image ID.
    Typically this function loads the image from a file, but
    in this case it generates the image on the fly from the
    specs in image_info.
    """
    image = Image.fromarray(self.load_image(idx))
    masks, labels, boxes = self.load_mask(idx)
    
    # create a BoxList from the boxes
    boxlist = BoxList(boxes, image.size, mode="xyxy")

    # add the labels to the boxlist
    boxlist.add_field("labels", torch.tensor(labels))

    # Add masks to the boxlist
    masks = np.transpose(masks, (2,0,1))
    masks = SegmentationMask(torch.tensor(masks), image.size, "mask")
    boxlist.add_field("masks", masks)
    
    # Important line! don't forget to add this
    if self.transforms:
        image, boxlist = self.transforms(image, boxlist)

    # return the image, the boxlist and the idx in your dataset
    return image, boxlist, idx
  
  
  def __len__(self):
      return self.num_examples
    

  def get_img_info(self, idx):
      # get img_height and img_width. This is used if
      # we want to split the batches according to the aspect ratio
      # of the image, as it can be more efficient than loading the
      # image from disk

      return {"height": self.height, "width": self.width}
    
  def get_gt(self):
      # Prepares dataset for coco eval
      
      
      images = []
      annotations = []
      results = []
      
      # Define categories
      categories = [ {"id": 1, "name": "square"}, {"id": 2, "name": "circle"}, {"id": 3, "name": "triangle"}]


      i = 1
      ann_id = 0

      for img_id, d in enumerate(self.image_info):
        import pdb;pdb.set_trace()

        images.append( {"id": img_id, 'height': self.height, 'width': self.width } )

        for (shape, color, dims) in d['shapes']:
          
          if shape == "square":
            category_id = 1
          elif shape == "circle":
            category_id = 2
          elif shape == "triangle":
            category_id = 3
          
          x, y, s = dims
          bbox = [ x - s, y - s, x+s, y +s ] 
          area = (bbox[0] - bbox[2]) * (bbox[1] - bbox[3])
          
          # Format for COCO
          annotations.append( {
              "id": int(ann_id),
              "category_id": category_id,
              "image_id": int(img_id),
              "area" : float(area),
              "bbox": [ float(bbox[0]), float(bbox[1]), float(bbox[2]) - float(bbox[0]) + 1, float(bbox[3]) - float(bbox[1]) + 1 ], # note that the bboxes are in x, y , width, height format
              "iscrowd" : 0
          } )

          ann_id += 1

      # Save ground truth file
      
      with open("tmp_gt.json", "w") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories }, f)

      # Load gt for coco eval
      self.coco = COCO("tmp_gt.json") 
      