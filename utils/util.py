#import tensorflow as tf
import os, cv2, math
import numpy as np
from lxml import etree
from PIL import Image, ImageDraw, ImageFont

HRSC_CLASSES = ['100000001', '100000002', '100000003', '100000004', 
               '100000005', '100000006', '100000007', '100000008', 
               '100000009', '100000010', '100000011', '100000012', 
               '100000013', '100000015', '100000016', '100000017', 
               '100000018', '100000019', '100000020', '100000022', 
               '100000024', '100000025', '100000026', '100000027', 
               '100000028', '100000029', '100000030', '100000032']

COLORS = [(166, 62, 97), (241, 214, 87), (214, 56, 119), (62, 80, 252), 
         (244, 189, 49), (183, 34, 118), (167, 225, 207), (61, 255, 183), 
         (139, 232, 255), (240, 219, 249), (253, 175, 67), (218, 76, 98), 
         (172, 132, 107), (63, 131, 220), (32, 39, 223), (228, 132, 173), 
         (158, 179, 22), (123, 14, 206), (97, 186, 127), (153, 170, 90), 
         (9, 88, 194), (157, 187, 231), (193, 106, 46), (243, 153, 47), 
         (176, 30, 0), (186, 212, 207), (37, 229, 16), (227, 26, 92)]


DOTA_CLASSES = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 
                'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 
                'roundabout', 'harbor', 'swimming-pool', 'helicopter' ] #, 'container-crane']

def read_examples_list(path):
    """Read list of training or validation examples.
    The file is assumed to contain a single example per line where the first
    token in the line is an identifier that allows us to find the image and
    annotation xml for that example.
    For example, the line:
    xyz 3
    would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).
    Args:
    path: absolute path to examples list file.
    Returns:
    list of example identifiers (strings).
    """
    with tf.gfile.GFile(path) as fid:
        lines = fid.readlines()

    return [line.strip().split(' ')[0] for line in lines]


def recursive_parse_xml_to_dict(xml):
  """Recursively parses XML contents to python dict.
  We assume that `object` tags are the only ones that can appear
  multiple times at the same level of a tree.
  Args:
    xml: xml tree obtained by parsing XML file contents using lxml.etree
  Returns:
    Python dictionary holding XML contents.
  """
  if not xml:
    return {xml.tag: xml.text}
  result = {}
  for child in xml:
    child_result = recursive_parse_xml_to_dict(child)
    if child.tag != 'object':
      result[child.tag] = child_result[child.tag]
    else:
      if child.tag not in result:
        result[child.tag] = []
      result[child.tag].append(child_result[child.tag])
  return {xml.tag: result}


def get_label_map_dict(label_map_path):
    """
    Read in dataset category name vs id mapping
    Args:
        xml file path which containing category name and ip information
    returns:
        Dict containing name to id mapping
    """
    tree = etree.parse(open(label_map_path, "r"))
    name_id_mapping = {}
    for node in tree.xpath("category"):
        cate_name = node.findtext("name")
        cate_id = node.findtext("id")
        name_id_mapping[cate_name] = cate_id
    return name_id_mapping


def read_data(data, image_dir):
    img_path = os.path.splitext(os.path.join(image_dir, data['filename']))[0] + ".jpg"
    
    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        difficult_obj.append(int(difficult))

        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        class_name = obj['name']
        classes_text.append(class_name)
        classes.append(label_to_integer[class_name])
        
    return {"image":img_path, "width":width, "height":height, 
            "xmin":xmin, "ymin":ymin, "xmax":xmax, "ymax":ymax, 
            "classes":classes, "classes_text":classes_text, "difficult":difficult_obj}

def get_parsing(path, image_dir):
    
    with tf.gfile.GFile(path, 'r') as fid:
        xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = recursive_parse_xml_to_dict(xml)['annotation']
        
        return read_data(data, image_dir)
    
#=================================================================================================================================

def copy_script(logs_path):
    import shutil
    
    script_path = os.path.join(logs_path, "script/")
    
    if not os.path.exists(script_path):
        os.mkdir(script_path)
        
    py_files = [f for f in os.listdir() if ".py" in f]

    for py in py_files:
        shutil.copy(py, script_path)

    if os.path.exists(script_path+"basenet"):
        shutil.rmtree(script_path+"basenet")
    shutil.copytree("basenet", script_path+"basenet")


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    
    h = h / h.max()
    
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    
    return h


def tricube_kernel(size, factor=7):
    s = (size-1)/2
    m, n = [(ss - 1.) / 2. for ss in [size, size]]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    
    h_x = (1 - (abs((x/s)**3)))**factor  # tricube
    h_y = (1 - (abs((y/s)**3)))**factor  # tricube
    
    h = h_y.dot(h_x)
    
    h /= h.max()
    
    return h



def color_map(mask):
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask = mask[:, :, ::-1]
    
    return mask


def draw_boxes(img, bboxes, classes, difficult, COCO=False):
    if len(bboxes) == 0:
        return img

    #height, width, _ = img.shape
    width, height, _ = img.shape
    image = Image.fromarray(img)
    #image = img
    font = ImageFont.truetype(
        font='/data/DB/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] / 2+ 0.4).astype('int32'))

    thickness = (image.size[0] + image.size[1]) // 600
    draw = ImageDraw.Draw(image)

    for box, category, diff in zip(bboxes, classes, difficult):
        
        color = np.array([0, 255, 0]) if diff==0 else np.array([255, 255, 0])
        
        y1, x1, y2, x2 = [int(i) for i in box]

        p1 = (x1, y1)
        p2 = (x2, y2)

        category = VOC_CLASSES[category] if not COCO else COCO_CLASSES[int(category)]
        
        label = '{} {:.1f}%   '.format(category, 1)
        label_size = draw.textsize(label)
        text_origin = np.array([p1[0], p1[1] - label_size[1]])

        for i in range(thickness):
            draw.rectangle(
                [p1[0] + i, p1[1] + i, p2[0] - i, p2[1] - i],
                outline=tuple(color))

        draw.rectangle(
            [tuple(text_origin),
             tuple(text_origin + label_size)],
            fill=tuple(color))

        draw.text(
            tuple(text_origin),
            label, fill=(0, 0, 0),
            font=font)

    del draw
    return np.array(image)

def draw_boxes_v2(img, bboxes, classes, scores, COCO=False):
    if len(bboxes) == 0:
        return img

    #height, width, _ = img.shape
    width, height, _ = img.shape
    image = Image.fromarray(img)
    #image = img
    font = ImageFont.truetype(
        font='/data/DB/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] / 2 + 0.4).astype('int32'))

    thickness = (image.size[0] + image.size[1]) // 600
    draw = ImageDraw.Draw(image)

    for box, category, score in zip(bboxes, classes, scores):
        
        color = np.array([0, 255, 0])
        
        y1, x1, y2, x2 = [int(i) for i in box]

        p1 = (x1, y1)
        p2 = (x2, y2)

        #category = VOC_CLASSES[category] if not COCO else COCO_CLASSES[int(category)]
        
        #label = '{} {:.1f}%   '.format(category, score)
        label = ' %s  '%  category
        label_size = draw.textsize(label)
        text_origin = np.array([p1[0], p1[1] - label_size[1]])

        for i in range(thickness):
            draw.rectangle(
                [p1[0] + i, p1[1] + i, p2[0] - i, p2[1] - i],
                outline=tuple(color))

        draw.rectangle(
            [tuple(text_origin),
             tuple(text_origin + label_size)],
            fill=tuple(color))

        draw.text(
            tuple(text_origin),
            label, fill=(0, 0, 0),
            font=font)

    del draw
    return np.array(image)

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

    
def is_contain(label, labels):
    for l in labels:
        if label == l:
            return True
    return False
    

def make_border(mask, size):
    mask[0:3, :, :] = 128
    mask[size-3:size, :, :] = 128
    
    mask[:, 0:3, :] = 128
    mask[:, size-3:size, :] = 128
    
    
    return mask
            

def load_checkpoint(model, model_ckpt, strict=False):

    for i, j in model.named_parameters():
        
        #if (i in model_ckpt.keys() or strict) and 'hm_c' not in str(i) :
        if i in model_ckpt.keys():
            j.data = model_ckpt[i]
    
    
    
def tensor_rotate(x, angle, transpose=False):
    if transpose:
        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2)
        
    if angle == 90:
        x = x.transpose(2, 3).flip(3)
    elif angle == 180:
        x = x.flip(2).flip(3)  # 180
    elif angle == 270:
        x = x.transpose(2, 3).flip(2) 
        
    if transpose:
        # B, C, H, W -> B, H, W, C
        x = x.permute(0, 2, 3, 1)
        
    return x
    
    