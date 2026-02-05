from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import base64
import numpy as np
import csv
import sys
import argparse

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='D:\\geometric\\bottom_up_tsv', help='downloaded feature directory')
parser.add_argument('--output_dir', default='D:\\geometric\\mscoco\\feature', help='output feature files')

args = parser.parse_args()

csv.field_size_limit(100000000)  # Set an appropriate limit

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

infiles = [
    'karpathy_test_resnet101_faster_rcnn_genome.tsv',
    'karpathy_train_resnet101_faster_rcnn_genome.tsv.0',
    'karpathy_train_resnet101_faster_rcnn_genome.tsv.1',
    'karpathy_val_resnet101_faster_rcnn_genome.tsv'
]

os.makedirs(os.path.join(args.output_dir, 'up_down_100'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'up_down_100_fc'), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, 'up_down_100_box'), exist_ok=True)

for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.b64decode(item[field]), dtype=np.float32).reshape(
                    (item['num_boxes'], -1))
            np.savez_compressed(os.path.join(args.output_dir, 'up_down_100', str(item['image_id'])), feat=item['features'])
            np.save(os.path.join(args.output_dir, 'up_down_100_fc', str(item['image_id'])), item['features'].mean(0))
            np.save(os.path.join(args.output_dir, 'up_down_100_box', str(item['image_id'])), item['boxes'])
