# For Processing single file in folder bottom_up_tsv.
import os
import base64
import numpy as np
import csv
import sys
import argparse

csv.field_size_limit(2 ** 31 - 1)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

def process_file(file_path, output_folder):
    count = 0
    with open(file_path, "r", encoding='utf-8') as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in reader:
            if count % 1000 == 0:
                print(count)
            count += 1

            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])
            item['num_boxes'] = int(item['num_boxes'])

            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.b64decode(item[field].encode('utf-8')),
                                            dtype=np.float32).reshape((item['num_boxes'], -1))

            image_id = item['image_id']
            feats = item['features']

            np.savez_compressed(os.path.join(output_folder, str(image_id)), feat=feats)

def main(args):
    if not os.path.exists(args.outfolder):
        os.makedirs(args.outfolder)

    if os.path.isdir(args.infeats):
        for file_name in os.listdir(args.infeats):
            if file_name.endswith(".tsv"):
                file_path = os.path.join(args.infeats, file_name)
                process_file(file_path, args.outfolder)
    else:
        process_file(args.infeats, args.outfolder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infeats', default='karpathy_train_resnet101_faster_rcnn_genome.tsv.0', help='image features folder or file')
    parser.add_argument('--outfolder', default='./mscoco/feature/up_down_10_100', help='output folder')
    args = parser.parse_args()
    main(args)