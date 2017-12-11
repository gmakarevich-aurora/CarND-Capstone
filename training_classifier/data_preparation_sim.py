import io
import os
import tensorflow as tf

from bs4 import BeautifulSoup
from utils import dataset_util

OUTPUT_PATH = "./output"

PATH_TO_GREEN_IMGS = "./images_for_training/green/"
PATH_TO_RED_IMGS = "./images_for_training/red/"
PATH_TO_YELLOW_IMGS = "./images_for_training/yellow/"
PATH_TO_XML = "./images_for_training/"

COLOR = ["green", "red", "yellow"]
LABEL_DICT = {
    "Green": 1,
    "Red": 2,
    "Yellow": 3,
    "off": 4}
LABEL_PATH_DICT = {
    "Green": PATH_TO_GREEN_IMGS,
    "Red": PATH_TO_RED_IMGS,
    "Yellow": PATH_TO_YELLOW_IMGS
}

# read xml file
def read_xml(XML_PATH):
    xml = ""
    with open(XML_PATH) as f:
        xml = f.readlines()
    
    return BeautifulSoup(''.join([line.strip('\t') for line in xml]), "lxml")

# get file name
def get_file_name(xml_info):
    return xml_info.find('filename').text

# get boxes from xml
def get_detected_objects(xml_info):
    return xml_info.find_all('object')

# get image size
def get_img_size(xml_info):
    return (int(xml_info.size.width.text), int(xml_info.size.height.text))

# get detected object's color
def get_object_color(obj):
    return obj.find('name').text

# get bounds from boxes
def get_boxbounds(obj):
    xmin = int(obj.bndbox.xmin.text)
    ymin = int(obj.bndbox.ymin.text)
    xmax = int(obj.bndbox.xmax.text)
    ymax = int(obj.bndbox.ymax.text)
    return (xmin, ymin, xmax, ymax)

# convert to TFRecords format
def create_tf_xml(XML_Path):
    # read xml info
    xml_info = read_xml(XML_Path)
    
    # get image info
    file_name = get_file_name(xml_info)
    detected_objects = get_detected_objects(xml_info)
    color = get_object_color(detected_objects[0]) # get color to find the correct folder path
    image_path = os.path.join(LABEL_PATH_DICT[color], file_name)

    # get image size
    img_size = get_img_size(xml_info)
    height = img_size[1] # Image height
    width = img_size[0] # Image width
    
    # open image
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()
    
    # encode
    encoded_image_io = io.BytesIO(encoded_image)
    filename = file_name.encode()
    image_format = 'png'.encode()
    
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
    # loop through each detection
    for box in detected_objects:
        # get detection info
        color = get_object_color(box)
        bounds = get_boxbounds(box)
        
        xmins.append(float(bounds[0]/width))
        ymins.append(float(bounds[1]/height))
        xmaxs.append(float(bounds[2]/width))
        ymaxs.append(float(bounds[3]/height))
        
        classes_text.append(color.encode())
        classes.append(int(LABEL_DICT[color]))
    
    print ('xmins: %s' % xmins)
    print ('ymins: %s' % ymins)
    print ('xmaxs: %s' % xmaxs)
    print ('ymaxs: %s' % ymaxs)
    print('classes: %s' % classes)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def do_conversion(path_to_xml):
    # fetch all XML files
    xml_list = []
    for fname in os.listdir(path_to_xml):
        if os.path.splitext(fname)[1] == '.xml':
            xml_list.append(fname)
    
    # TF Records file name
    xml2records_filename = 'xml_train.tfrecords'
    
    # Create TFRecords file
    writer = tf.python_io.TFRecordWriter(os.path.join(OUTPUT_PATH, xml2records_filename))
    
    for xml_filename in xml_list:
        xml_path = path_to_xml + xml_filename
        tf_example = create_tf_xml(xml_path)
        writer.write(tf_example.SerializeToString())
    
    writer.close()

if __name__ == '__main__':
    do_conversion(PATH_TO_XML)
