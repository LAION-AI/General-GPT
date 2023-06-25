'''
Adapted from: https://github.com/allenai/mmc4/blob/main/scripts/compute_assignments.py

example usage:
python compute_assignment.py docs_shard_{$SHARD}_v2.jsonl
'''
import argparse
import json
import os
import numpy as np
import linear_assignment
from tqdm import tqdm
import braceexpand
import webdataset as wds
from itertools import islice
import boto3
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_shards', default="{000000..000500}.tar")
    return parser.parse_args()


def get_image_assignments(im2txt):
    '''
    returns a list assignments of length N_images such that assignments[i] is the sentence index that image i was assigned to.
    '''
    im_idxs_s, txt_idxs_s, sol = linear_assignment.base_solve(-im2txt)
    im2txt_idxs = {im_idxs_s[k]: txt_idxs_s[k] for k in range(len(im_idxs_s))}
    if im2txt.shape[0] > im2txt.shape[1]:
        # there are more images than sentences. we dont want to discard images. so, for unassigned images, we will put them with their corresponding max.
        for imidx in range(len(im2txt)):
            if imidx not in im2txt_idxs:
                im2txt_idxs[imidx] = int(np.argmax(im2txt[imidx]))

    return [im2txt_idxs[idx] for idx in range(len(im2txt_idxs))]


def augment_assignments(data):
    data = json.loads(data)
    image_info = data['image_info']
    im2txt = np.array(data['similarity_matrix'])
    assignment = get_image_assignments(im2txt)

    for im_idx, im in enumerate(image_info):
        im['matched_text_index'] = int(assignment[im_idx])
        im['matched_sim'] = float(im2txt[im_idx, assignment[im_idx]])
    
    return data

def main():
    args = parse_args()

    shards = braceexpand.braceexpand(args.input_shards)
    
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket('s-laion')

    # with Pool(30) as pool:
    for s in shards:
        print(f"Augmenting {s} ...")
        dataset = wds.WebDataset(f"pipe:aws s3 cp s3://s-laion/mmc4/mmc4_wds/{s} -").decode(
                wds.handle_extension("json", augment_assignments)
            ).to_tuple("__key__", "__url__", "json")

        with wds.TarWriter(s) as dst:
            for key, url, js in islice(dataset, 0, 999999999):
                sample = {
                    "__key__": key,
                    "__url__": url,
                    "json": js
                }
                dst.write(sample)

        my_bucket.upload_file(os.path.join("/fsx/home-shivr", s), os.path.join("mmc4/mmc4_wds", s))
        os.remove(os.path.join("/fsx/home-shivr", s))


if __name__ == '__main__':
    main()