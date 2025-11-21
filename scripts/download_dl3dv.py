""" This script is used to download the DL3DV benchmark from the huggingface repo.

    The benchmark is composed of 140 different scenes covering different scene complexities (reflection, transparency, indoor/outdoor, etc.) 

    The whole benchmark is very large: 2.1 TB. So we provide this script to download the subset of the dataset based on common needs. 


        - [x] Full benchmark downloading
            Full download can directly be done by git clone (w. lfs installed).

        - [x] scene downloading based on scene hash code  

        Option: 
        - [x] images_4 (960 x 540 resolution) level dataset (approx 50G)

"""
# TO download sspecific scenes from DL3DV dataset:
# Note, it is suggested to use --clean_cache flag as it saves space by cleaning the cache folder created by huggingface hub API. 
# e.g. a scene with hash 0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695
# python download.py  --subset hash --hash 0853979305f7ecb80bd8fc2c8df916410d471ef04ed5f1a64e9651baa41d7695 --odir /w/20251/alvarolopez/datasets --only_level8 
# python download.py  --subset hash --hash 9641a1ed7963ce5ca734cff3e6ccea3dfa8bcb0b0a3ff78f65d32a080de2d71e --odir /w/20251/alvarolopez/datasets --only_level8
# Note: takes like 40 minutes to download on 10 mb/s download speed for EACH scene.
# Note, youll need to add your Huggingface token inside the script to be able to download the dataset.
#hashes of individual scenes can be found in: https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/blob/main/benchmark-meta.csv

# then: run python src/scripts/convert_dl3dv_test.py     --input_dir /w/20251/alvarolopez/datasets     --output_dir datasets/dl3dv/test     --img_subdir images_8     --n_test 1 
# this will generate the test data for that one single scene, which is gonna be one torch file per scene
# then we need to modify the generate_dl3dv_index.py to ahve the path to our dataset, in this case: DATASET_PATH = Path("/u/alvarolopez/Documents/csc2529/depthsplat/depthsplat/datasets/dl3dv/")
# Then you need to do wget to donwload the model on teh pretrained dir: wget -P pretrained https://huggingface.co/haofeixu/depthsplat/resolve/main/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth
# Then run: CUDA_VISIBLE_DEVICES=0 python -m src.main +experiment=dl3dv     dataset.test_chunk_interval=1     dataset.roots=[datasets/dl3dv]     dataset.image_shape=[256,448]     dataset.ori_image_shape=[270,480]     model.encoder.num_scales=2     model.encoder.upsample_factor=4     model.encoder.lowest_feature_resolution=8     model.encoder.monodepth_vit_type=vitb     checkpointing.pretrained_model=pretrained/depthsplat-gs-base-dl3dv-256x448-randview2-6-02c7b19d.pth     mode=test     dataset/view_sampler=evaluation     dataset.view_sampler.num_context_views=6     dataset.view_sampler.index_path=assets/dl3dv_start_0_distance_50_ctx_6v_video_0_50.json     test.save_image=true     test.save_depth=true     test.save_depth_npy=true     test.save_gaussian=true     output_dir=outputs/dl3dv_single_scene
# NOTE: THIS IS USING ONE OF THE PREEXISTING INDEXES, MAYBGE WE COULD CREATE OUR OWN?


import os 
from os.path import join
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi 
import argparse
import traceback
import pickle
import shutil
import dotenv
from huggingface_hub import login

dotenv.load_dotenv()

login(token=os.getenv('HUGGINGFACE_TOKEN'))

api = HfApi()
repo_root = 'DL3DV/DL3DV-10K-Benchmark'


def hf_download_path(repo_path: str, odir: str, max_try: int = 5):
    """ hf api is not reliable, retry when failed with max tries

    :param repo_path: The path of the repo to download
    :param odir: output path 
    """	
    rel_path = os.path.relpath(repo_path, repo_root)

    counter = 0
    while True:
        if counter >= max_try:
            print("ERROR: Download {} failed.".format(repo_path))
            return False

        try:
            api.hf_hub_download(repo_id=repo_root, filename=rel_path, repo_type='dataset', local_dir=odir, cache_dir=join(odir, '.cache'))
            return True

        except BaseException as e:
            traceback.print_exc()
            counter += 1
            print(f'Retry {counter}')
    

def clean_huggingface_cache(cache_dir: str):
    """ Huggingface cache may take too much space, we clean the cache to save space if necessary

    :param cache_dir: the current cache directory 
    """    
    # Current huggingface hub does not provide good practice to clean the space.  
    # We mannually clean the cache directory if necessary. 
    shutil.rmtree(join(cache_dir, 'datasets--DL3DV--DL3DV-10K-Benchmark'))


def download_by_hash(filepath_dict: dict, odir: str, hash: str, only_level4: bool, only_level8: bool = False):
    """ Given a hash, download the relevant data from the huggingface repo 

    :param filepath_dict: the cache dict that stores all the file relative paths 
    :param odir: the download directory 
    :param hash: the hash code for the scene 
    :param only_level4: the images_4 resolution level, if true, only the images_4 resolution level will be downloaded 
    """	
    all_files = filepath_dict[hash]
    download_files = [join(repo_root, f) for f in all_files] 

    if only_level4: # only download images_4 level data
        download_files = []
        for f in all_files:
            subdirname = os.path.basename(os.path.dirname(f))

            if 'images' in f and subdirname != 'images_4' or 'input' in f:
                continue 

            download_files.append(join(repo_root, f))
            
    if only_level8: # only download images_8 level data
        download_files = []
        for f in all_files:
            subdirname = os.path.basename(os.path.dirname(f))

            if 'images' in f and subdirname != 'images_8' or 'input' in f:
                continue 

            download_files.append(join(repo_root, f))

    for f in download_files:
        if hf_download_path(f, odir) == False:
            return False

    return True
    

def download_benchmark(args):
    """ Download the benchmark based on the user inputs.

        1. download the benchmark-meta.csv
        2. based on the args, download the specific subset 
            a. full benchmark 
            b. full benchmark in images_4 resolution level 
            c. full benchmark only with nerfstudio colmaps (w.o. gaussian splatting colmaps) 
            d. specific scene based on the index in [0, 140)

    :param args: argparse args. Used to decide the subset.
    :return: download success or not
    """	
    output_dir = args.odir
    subset_opt = args.subset
    level4_opt = args.only_level4
    level8_opt = args.only_level8
    hash_name  = args.hash
    is_clean_cache = args.clean_cache

    # import pdb; pdb.set_trace()
    os.makedirs(output_dir, exist_ok=True)

    # STEP 1: download the benchmark-meta.csv and .cache/filelist.bin
    meta_repo_path = join(repo_root, 'benchmark-meta.csv')
    cache_file_path = join(repo_root, '.cache/filelist.bin')
    if hf_download_path(meta_repo_path, output_dir) == False:
        print('ERROR: Download benchmark-meta.csv failed.')
        return False

    if hf_download_path(cache_file_path, output_dir) == False:
        print('ERROR: Download .cache/filelist.bin failed.')
        return False


    # STEP 2: download the specific subset
    df = pd.read_csv(join(output_dir, 'benchmark-meta.csv'))
    filepath_dict = pickle.load(open(join(output_dir, '.cache/filelist.bin'), 'rb'))
    hashlist = df['hash'].tolist()
    download_list = hashlist

    # sanity check 
    if subset_opt == 'hash':  
        if hash_name not in hashlist: 
            print(f'ERROR: hash {hash_name} not in the benchmark-meta.csv')
            return False

        # if subset is hash, only download the specific hash
        download_list = [hash_name]

    
    # download the dataset 
    for cur_hash in tqdm(download_list):
        if download_by_hash(filepath_dict, output_dir, cur_hash, level4_opt, level8_opt) == False:
            return False

        if is_clean_cache:
            clean_huggingface_cache(join(output_dir, '.cache'))

    return True 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--odir', type=str, help='output directory', default='DL3DV-10K-Benchmark')
    parser.add_argument('--subset', choices=['full', 'hash'], help='The subset of the benchmark to download', required=True)
    parser.add_argument('--only_level4', action='store_true', help='If set, only the images_4 resolution level will be downloaded to save space')
    parser.add_argument('--only_level8', action='store_true', help='If set, only the images_8 resolution level will be downloaded to save space')
    parser.add_argument('--clean_cache', action='store_true', help='If set, will clean the huggingface cache to save space')
    parser.add_argument('--hash', type=str, help='If set subset=hash, this is the hash code of the scene to download', default='')
    params = parser.parse_args()


    if download_benchmark(params):
        print('Download Done. Refer to', params.odir)
    else:
        print(f'Download to {params.odir} Failed. See error messsage.')

