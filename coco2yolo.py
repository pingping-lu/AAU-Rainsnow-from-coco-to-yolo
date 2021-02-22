import json
import argparse
import funcy
import os
import shutil
from sklearn.model_selection import train_test_split
import pdb
import numpy as np

parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('--root', type=str, help='Where to store COCO images')
parser.add_argument('--annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('--s', dest='split', type=float, required=False, default=1,
                    help="A percentage of a split for train and val; a number in (0, 1), if s==1, val==test")
parser.add_argument('--having-annotations', dest='having_annotations', action='store_true',
                    help='Ignore all images without annotations. Keep only these with at least one annotation')
parser.add_argument('--coco_category', dest='coco_category', action='store_true',
                    help='using coco class name')
parser.add_argument('--label_only', dest='label_only', action='store_true',
                    help='only modify label folder')

args = parser.parse_args()

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

all_set = ['Egensevej-1','Egensevej-2','Egensevej-3','Egensevej-4','Egensevej-5',
            'Hadsundvej-1','Hadsundvej-2',
            'Hasserisvej-1','Hasserisvej-2','Hasserisvej-3',
            'Hjorringvej-1','Hjorringvej-2','Hjorringvej-3','Hjorringvej-4',
            'Hobrovej-1',
            'Ostre-1','Ostre-2','Ostre-3','Ostre-4',
            'Ringvej-1','Ringvej-2','Ringvej-3'
            ]

test_set = ['Egensevej-2','Ostre-4','Ringvej-3'] #snow night rain

exclude_set = ['Hadsundvej-1','Hadsundvej-2'] #snow night rain

select_classes = ['person', 'bicycle', 'car', 'bus', 'truck']

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)

def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)

def main(args):
    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info']
        licenses = coco['licenses']
        images = coco['images']
        annotations = coco['annotations']
        for item in coco["categories"]:
            item['name'] = CLASSES[int(item['id'])-1]
        categories = coco['categories']

        annotations = [item for item in annotations if item['area']>0]

        number_of_images = len(images)

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        if args.having_annotations:
            images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        x = [item for item in images if item['file_name'].split('/')[1] not in (test_set + exclude_set)] 
        y = [item for item in images if item['file_name'].split('/')[1] in test_set]
        if args.split>0 and args.split<1:
            x_train, x_val = train_test_split(x, train_size=args.split)
        else:
            x_val = y.copy()
            x_train = x.copy()

        if not args.coco_category:
            category_map = {item['id']:select_classes.index(item['name']) for item in categories if item['name'] in select_classes}
        else:
            category_map = {item['id']:int(item['id']-1) for item in categories if item['name'] in select_classes}

        root = args.root
        label_folder = os.path.join(root,'labels')
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        # else:
        #     print('delete {} ...'.format(label_folder))
        #     shutil.rmtree(label_folder)
        #     os.makedirs(label_folder)

        image_folder = os.path.join(root,'images')
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)


        lists = [x_train, x_val, y]
        lists_mode = ['train','val','test']
        if 'thermal' in args.annotations:
            suffix = '_thermal'
        else:
            suffix = '_rgb'
        lists_mode = [item+suffix for item in lists_mode]

        for items,mode in zip(lists,lists_mode):

            if not os.path.exists(os.path.join(label_folder,mode)):
                os.makedirs(os.path.join(label_folder,mode))

            if not os.path.exists(os.path.join(image_folder,mode)):
                os.makedirs(os.path.join(image_folder,mode))

            for item in items:
                txt_name = os.path.join(label_folder,mode,item['file_name'].replace('png','txt').replace('/','_'))
                
                if not args.label_only:
                    image_name = os.path.join(image_folder,mode,item['file_name'].replace('/','_'))
                    shutil.copyfile(os.path.join(root,item['file_name']),image_name) 

                # item['file_name'] = item['file_name'].replace('/','_')
                anns = funcy.lfilter(lambda a: int(a['image_id']) in [item['id']], annotations)
                fid = open(txt_name,'w')
                for ann in anns:
                    if ann['category_id'] in category_map:
                        bbox = ann['bbox']
                        bbox[0] = np.max([0.,bbox[0]])
                        bbox[1] = np.max([0.,bbox[1]])
                        bbox[2] = np.min([bbox[0] + bbox[2], item['width']-1]) - bbox[0]
                        bbox[3] = np.min([bbox[1] + bbox[3], item['height']-1]) - bbox[1]
                        if bbox[2]*bbox[3]>0:
                            fid.write('%d %f %f %f %f\n'%(
                                category_map[ann['category_id']],
                                (bbox[0]+bbox[2]/2.0)/item['width'],
                                (bbox[1]+bbox[3]/2.0)/item['height'],
                                bbox[2]/item['width'],
                                bbox[3]/item['height']))
                fid.close()


        print("Saved {} entries in train {} in val, and {} in test".format(len(x_train), len(x_val), len(y)))


if __name__ == "__main__":
    main(args)