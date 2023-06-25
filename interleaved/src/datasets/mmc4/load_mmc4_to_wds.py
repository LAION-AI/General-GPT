"""
Adapted from: https://github.com/allenai/mmc4/blob/main/scripts/download_images.py
"""

import argparse
import json
import os
import io
import uuid
import zipfile
import urllib
import requests
import shutil
import base64
from tqdm import tqdm
from PIL import Image
import tqdm
import magic
import time
import shelve
import pandas as pd
import glob

import braceexpand
import webdataset as wds
from itertools import islice
from multiprocessing import Pool, Queue, cpu_count
import boto3


headers = {
    'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--output_image_dir",
    default="mmc4_images",
    type=str
)
arg_parser.add_argument(
    "--shards",
    default="https://storage.googleapis.com/ai2-jackh-mmc4-public/data/docs_no_face_shard_{0..23098}_v2.jsonl.zip",
    type=str,
    help="Pass in a list of shards in the format https://..._{0..23098}_v2.jsonl.zip",
)
arg_parser.add_argument(
    "--post_fetch",
    default=False,
    action="store_true",
    help="Whether to simply augment data with image bytes"
)
arg_parser.add_argument('--num_process', type=int, default=64, help='Number of processes in the pool can be larger than cores')
arg_parser.add_argument('--chunk_size', type=int, default=100, help='Number of images per chunk per process')
arg_parser.add_argument('--shard_name', type=str, default=None)
arg_parser.add_argument('--report_dir', type=str, default='./mmc4_images_status_report/', help='Local path to the directory that stores the downloading status')


args = arg_parser.parse_args()


def gather_image_info_shard(json_file):
    """Gather image info from shard"""
    samples = []
    data = []
    for sample_data in tqdm.tqdm(json_file):
        # get image names from json
        sample_data = json.loads(sample_data)
        samples.append(sample_data)
        for img_item in sample_data['image_info']:
            data.append({
                'local_identifier': img_item['image_name'],
                'url': img_item['raw_url'],
            })
    return samples, data


def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)


def download_image(row):
    fname = row['folder'] + '/' + row['local_identifier']

    # Skip already downloaded images, retry others later
    if os.path.isfile(fname):
        row['status'] = 200
        row['file'] = fname
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
        return row

    try:
        # Use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        rate_limit_idx = 0
        while response.status_code == 429:
            print(f'RATE LIMIT {rate_limit_idx} for {row["local_identifier"]}, will try again in 2s')
            response = requests.get(row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
            row['status'] = response.status_code
            rate_limit_idx += 1
            time.sleep(2)
            if rate_limit_idx == 5:
                print(f'Reached rate limit for {row["local_identifier"]} ({row["url"]}). Will skip this image for now.')
                row['status'] = 429
                return row

    except Exception as e:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row

    if response.ok:
        try:
            with open(fname, 'wb') as out_file:
                # some sites respond with gzip transport encoding
                response.raw.decode_content = True
                out_file.write(response.content)

            # Resize image if it is too big
            # call('mogrify -resize "800x800>" {}'.format(fname))
            img = Image.open(fname)
            if max(img.size) > 800:
                img = img.resize((min(img.width, 800), min(img.height, 800)))
                img.save(fname)

            row['mimetype'] = magic.from_file(fname, mime=True)
            row['size'] = os.stat(fname).st_size
        except:
            # This is if it times out during a download or decode
            row['status'] = 408
            return row
        row['file'] = fname
    return row


def save_status(args, shelve_filename):
    print(f'Generating Dataframe from results...')
    with shelve.open(shelve_filename) as results:
        keylist = sorted([int(k) for k in results.keys()])
        df = pd.concat([results[str(k)][1] for k in keylist], sort=True)
    
    report_filename = os.path.join(args.report_dir, f'{args.shard_name}.tsv.gz')
    df.to_csv(report_filename, sep='\t', compression='gzip', header=False, index=False)
    print(f'Status report saved to {report_filename}')
    
    print('Cleaning up...')
    matched_files = glob.glob(f'{shelve_filename}*')
    for fn in matched_files:
        os.remove(fn)


def download_images_multiprocess(args, df, func):
    """Download images with multiprocessing"""

    chunk_size = args.chunk_size
    num_process = args.num_process

    print('Generating parts...')

    shelve_filename = '%s_%s_%s_results.tmp' % (args.shard_name, func.__name__, chunk_size)
    with shelve.open(shelve_filename) as results:

        pbar = tqdm.tqdm(total=len(df), position=0)
        # Resume:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "Resuming"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = ((index, df[i:i + chunk_size], func) for index, i in enumerate(range(0, len(df), chunk_size)) if index not in finished_chunks)
        pbar.write(f'\t{int(len(df) / chunk_size)} parts. Using {num_process} processes.')

        pbar.desc = "Downloading"
        with Pool(num_process) as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 20)):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    return shelve_filename


def convert_to_bytes(input_args):
    output_image_dir, sample_data, idx = input_args
    image_info = sample_data["image_info"]

    #Add each image to the tar file
    for info in image_info:
        fname = f"{output_image_dir}/{idx}/{info['image_name']}"
        if os.path.exists(fname) and 'image_bytes' not in info:
            try:
                img = Image.open(fname)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                info['image_bytes'] = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
                info['valid'] = str(1)
            except:
                info['valid'] = str(0)
                continue
        else:
            info['valid'] = str(1)
    
    return sample_data


def main():
    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    doc_shards = list(braceexpand.braceexpand(args.shards))
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket('s-laion')
    existing_files = [i.key.split("/")[-1] for i in my_bucket.objects.filter(Prefix='mmc4/mmc4_wds') if i.key.endswith('.tar')]

    skip_count = 0

    for idx in range(len(doc_shards)):
        print("Attempting to download zip for shard", idx)
        try:
            output_file = os.path.join("/fsx/home-shivr/mmc4_docs","%06d.tar"%idx)
            if output_file.split("/")[-1] in existing_files:
                print("Shard already exists, moving to next.")
                continue

            urllib.request.urlretrieve(doc_shards[idx], "temp.zip")
            samples = []
            # Open the ZIP archive and extract the JSON file
            with zipfile.ZipFile("temp.zip", "r") as zip_file:
                # Assumes the JSON file is the first file in the archive
                json_filename = zip_file.namelist()[0]
                with zip_file.open(json_filename, "r") as json_file:
                    samples, data = gather_image_info_shard(json_file)

                    shard_folder = args.output_image_dir + "/" + str(idx)
                    if not os.path.exists(shard_folder):
                        os.makedirs(shard_folder)
                    
                    for d in data:
                        d['folder'] = shard_folder

                    df = pd.DataFrame(data)

                    args.shard_name = idx

                    # Download images
                    shelve_filename = download_images_multiprocess(
                        args=args, 
                        df=df,
                        func=download_image,
                    )
                    
                    # Save status & cleaning up
                    save_status(
                        args=args,
                        shelve_filename=shelve_filename,
                    )

            with wds.TarWriter(output_file) as sink:
                print("Converting saved images to bytes and saving...")
                pool_data = ((args.output_image_dir, sample_data, idx) for sample_data in samples)
                with Pool(args.num_process) as pool:
                    for out_json in pool.imap_unordered(convert_to_bytes, pool_data, 20):
                        key_str = "%s-%s"%(idx, uuid.uuid4().hex)
                        result = {"__key__": key_str, "json": out_json}
                        sink.write(result)
            
            # upload files to s3 and remove local copies
            my_bucket.upload_file(os.path.join("/fsx/home-shivr/mmc4_docs", "%06d.tar"%idx), os.path.join("mmc4/mmc4_wds", "%06d.tar"%idx))
            os.remove(os.path.join("/fsx/home-shivr/mmc4_docs", "%06d.tar"%idx))
            shutil.rmtree(os.path.join(args.output_image_dir, str(idx)))
            os.remove("temp.zip")

        except urllib.error.HTTPError as e:
            print("Skipping shard", idx)
            skip_count += 1
            continue
    
    print("Total Skipped =", skip_count)


if __name__ == "__main__":
    main()