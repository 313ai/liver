import tqdm
import cv2
import openslide
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import glob
import os.path
import os
import sys
import shutil
from collections import defaultdict

def pull_samples(slide_fn, level, tile_size=256, progress=tqdm.tqdm):
    print(slide_fn)
    slide = openslide.open_slide(slide_fn)

    keep = []
    empty = []
    bad = []
    tiles = []
    zooms = []

    min_empty=0.0
    max_empty=0.50
    min_std=2.
    slide_dim = slide.level_dimensions[level]
    slide_dim_0 = slide.level_dimensions[0]


    slide_width, slide_height = slide_dim
    slide_width_0, slide_height_0 = slide_dim_0

    tile_rgba = np.array(slide.read_region((0,0),level,slide_dim))
    gray = cv2.cvtColor(tile_rgba,cv2.COLOR_BGR2GRAY)
    _, 
    
    h = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    pct_e = []
    tile_range = list(range(0,slide_height, tile_size))
    for row in progress(tile_range):
        for col in range(0, slide_width, tile_size):
            tile_width = min(tile_size, slide_width - col)
            tile_height = min(tile_size, slide_height - row)

            tile_loc_0 = ( col * slide_width_0 // slide_width, row * slide_height_0 // slide_height  )
            tile = slide.read_region(tile_loc_0, level, (tile_width, tile_height))

            thresh_area = thresh[row:(row+tile_height), col:(col+tile_width)]
            area_sum = (thresh_area > -1).sum()
            if area_sum == 0:
                bad.append(tile)
            else:
                pct_empty = (thresh_area == 0).sum()/area_sum
                pct_e.append(pct_empty)
                if pct_empty < max_empty:
                    keep.append(tile)

    slide.close()
    return keep

def plot_slides(slide_array):
    thumb_size = (50,25)
    num_slides = 20
    slide_sample_idx = np.random.choice(range(len(slide_array)), size=num_slides)
    plt.figure(figsize=thumb_size)
    columns = 5
    for i, slide_idx in tqdm.tqdm(enumerate(slide_sample_idx), total=num_slides):
        plt.subplot(num_slides / columns + 1, columns, i + 1)
        plt.imshow(slide_array[slide_idx])

def plots_from_samples(slide_files, tumor_slides, level=1):
    rand_tumor_fn = tumor_slides[np.random.choice(range(len(tumor_slides)))].upper()
    print(rand_tumor_fn)
    keep, empty, zooms = pull_samples(slide_files[rand_tumor_fn], level)
    plot_slides(keep)
    plot_slides(zooms)

def build_samples(
    slide_path = "/DATA/BIO/GDC/liver/slides/",
    samples_path = "/DATA/BIO/GDC/liver/samples/",
    slides_path = "/DATA/BIO/GDC/liver/slides.csv",
    target_tile_size = 256,
    level = 1,
    progress=tqdm.tqdm
):
    slide_files = { os.path.basename(fn).upper(): fn for fn in glob.glob(slide_path+'*')}

    slide_df = pd.read_csv(slides_path)

    for slide_name, slide_fn in progress(slide_files.items()):
        img_dir = os.path.join(samples_path, slide_name)
        if not os.path.exists(img_dir):
            img_dir_1 = os.path.join(samples_path, slide_name, "level_1")
            if not os.path.exists(img_dir_1):
                os.makedirs(img_dir_1)
                
                keep_1 = pull_level_slides(
                    slide_fn, 
                    target_tile_size=target_tile_size, 
                    level=level, 
                    progress=progress)
                for i in progress(list(range(len(keep_1)))):
                    img_1_fn = os.path.join(img_dir_1,"%d.tiff" % i)
                    keep_1[i].save(img_1_fn)


def get_tissue_map(slide):
    thumb = slide.associated_images['thumbnail']
    tile_rgba = np.array(thumb)
    gray = cv2.cvtColor(tile_rgba,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh

def pull_zoom_samples(slide_fn, tile_size=256, progress=tqdm.tqdm):
    import pdb; pdb.set_trace()

    slide = openslide.open_slide(slide_fn)

    level_0_tiles = []
    level_1_tiles = []

    max_empty=0.50

    slide_dim_1 = slide.level_dimensions[1]
    slide_width_1, slide_height_1 = slide_dim_1

    slide_dim_0 = slide.level_dimensions[0]
    slide_width_0, slide_height_0 = slide_dim_0

    downsample_ratio = slide.level_downsamples[1]


    # level 1 fits in memory, lets use that to figure out where we have
    # actual tissue
    tile_rgba = np.array(slide.read_region((0,0),1,slide_dim_1))
    gray = cv2.cvtColor(tile_rgba,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    # loop throught the level 1 tiles, when we find one to keep, we'll
    # also pull the level 0 tiles
    tile_range = list(range(0,slide_height_1, tile_size))
    for row in progress(tile_range):
        for col in range(0, slide_width_1, tile_size):
            tile_width_1 = min(tile_size, slide_width_1 - col)
            tile_height_1 = min(tile_size, slide_height_1 - row)

            tile_loc_0 = ( col * slide_width_0 // slide_width_1, row * slide_height_0 // slide_height_1  )
            thresh_area = thresh[row:(row+tile_height_1), col:(col+tile_width_1)]
            area_sum = (thresh_area > -1).sum()
            if area_sum > 0:
                pct_empty = (thresh_area == 0).sum()/area_sum
                if pct_empty < max_empty:
                    tile = slide.read_region(tile_loc_0, 1, (tile_width_1, tile_height_1))
                    level_1_tiles.append(tile)
                    tile_width_0 = int(tile_width_1*downsample_ratio)
                    tile_height_0 = int(tile_height_1*downsample_ratio)
                    tile_0 = slide.read_region(tile_loc_0, 0, (tile_width_0, tile_height_0))
                    level_0_tiles.append(tile_0)
    slide.close()

    return level_1_tiles, level_0_tiles


def pull_level_slides(slide_fn, target_tile_size = 256, level=1, progress=tqdm.tqdm):
    max_empty=0.50
    slide = openslide.open_slide(slide_fn)
    tmap = get_tissue_map(slide)
    tmap_h, tmap_w = tmap.shape
    slide_w, slide_h = slide.level_dimensions[level]
    
    h_ratio = tmap_h / slide_h
    w_ratio = tmap_w / slide_w
    ratio = min(h_ratio, w_ratio)
    tile_size = int(ratio * target_tile_size)
    #print('tile size: ', tile_size)
    
    tile_range = list(range(0,tmap_h, tile_size))
    tiles = []

    for row in progress(tile_range):
        for col in range(0, tmap_w, tile_size):
            tile_width = min(tile_size, tmap_w - col)
            tile_height = min(tile_size, tmap_h - row)
            thresh_area = tmap[row:(row+tile_height), col:(col+tile_width)]
            area_sum = (thresh_area > -1).sum()
            if area_sum > 0:
                pct_empty = (thresh_area == 0).sum()/area_sum
                if pct_empty < max_empty:
                    def pull_tile(slide, level, zoom_size):
                        slide_w, slide_h = slide.level_dimensions[level]
                        slide_0_w, slide_0_h = slide.level_dimensions[0]
                        #zoom_size = int(tile_size * slide_w / tmap_w)
                        x = int(col * slide_0_w / tmap_w)
                        y = int(row * slide_0_h / tmap_h)
                        #print('zoom: ', zoom_size)
                        try:
                            tile = slide.read_region((x,y), level, (zoom_size, zoom_size))
                        except:
                            slide.close()
                            slide = openslide.open_slide(slide_fn)
                            tile = None

                        return slide, tile

                    slide, tile_1 = pull_tile(slide, level, target_tile_size)

                    if not tile_1 is None:
                        tiles.append(tile_1)

    slide.close()
    return tiles

def build_liver_csv_data(
        liver_path, exp_path, train_dir, test_dir, 
        csv_name='records.csv', samples='samples', progress=tqdm.tqdm,
        force_rebuild=False, split=0.8, val_split=0.8, samples_per_patient=30,
        slide_level='level_1'
    ):
    if force_rebuild:
        shutil.rmtree(str(exp_path))
    csv_path = exp_path/csv_name
    if not csv_path.exists():
        print("build traing/val/test csv data")

        slides = pd.read_csv(liver_path/'slides.csv')
        slides = slides.loc[slides.sample_type_id.isin([1])]
        slide_info = defaultdict(dict)

        def pull_tiles(slides, patient_id, num_tiles, slide_level):
            slide_fns = []
            tiles = []

            # get list of candidate samples
            for i, row in slides.loc[slides.submitter_id == patient_id].iterrows():
                slide_name = row.slide_file_name
                sfp = liver_path/samples/row.slide_file_name.upper()/slide_level
                slide_fns = list(sfp.iterdir())

            num_samples = len(slide_fns)
            tiles = list(np.random.choice(slide_fns, size=min(num_tiles,num_samples), replace=False))

            return tiles



        def build_tiles(patients, dsname, folder):
            records = []
            folder.mkdir(parents=True, exist_ok=True)
            for p in tqdm.tqdm_notebook(patients):
                tiles = pull_tiles(slides, p, samples_per_patient, slide_level)
                for i, tile_fn in enumerate(tiles):
                    base_name = '%s_%04d.tiff' % (p, i)
                    dest_tile = folder/base_name
                    os.symlink(tile_fn, dest_tile)
                    records.append({
                        'patient': p,
                        'dsname': dsname,
                        'event_time': int(slides.loc[slides.submitter_id == p, 'days_proxy'].iloc[0]),
                        'event_type': slides.loc[slides.submitter_id == p, 'event_observed'].iloc[0],
                        'src_tile': tile_fn,
                        'age_at_diagnosis': int(slides.loc[slides.submitter_id == p, 'age_at_diagnosis'].iloc[0]),
                        'percent_tumor_cells': int(slides.loc[slides.submitter_id == p, 'percent_tumor_cells'].iloc[0]),
                        'dest_tile': dest_tile
                    })
            return records



        # create event time, drop any nulls, create event observed
        slides['days_proxy'] = slides.days_to_death.fillna(slides.days_to_last_follow_up)
        slides = slides.loc[slides.days_proxy.notnull()]
        slides['event_observed'] = True
        slides.loc[slides.days_to_last_follow_up.notnull(),'event_observed'] = False    
        slides['event_observed'] = slides['event_observed'].astype(int)

        # filter tumor only
        slides = slides.loc[slides.sample_type_id == 1]

        # no null age
        slides = slides.loc[slides.age_at_diagnosis.notnull()]

        #create censor label

        patients = list(set(slides.submitter_id))
        num_patients = len(patients)
        train_val_split = int(split * num_patients)
        train_split = int(val_split * train_val_split)

        random_patients = np.random.permutation(patients)
        train_patients = random_patients[0:train_split]
        valid_patients = random_patients[train_split:train_val_split]
        test_patients = random_patients[train_val_split:]

        # convert days_proxy to int for softmax
        slides['days_proxy'] = slides.days_proxy.astype(int)


        # arrange the sample data
        train_records = build_tiles(train_patients, 'train', train_dir)
        valid_records = build_tiles(valid_patients, 'valid', train_dir)
        test_records = build_tiles(test_patients, 'test', test_dir)

        csv_data = pd.DataFrame(train_records + valid_records + test_records)
        csv_data.to_csv(csv_path, index=False)
    else:
        print("csv data already built")

    csv_data = pd.read_csv(csv_path)
    
    return csv_data