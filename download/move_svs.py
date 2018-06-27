import glob
import shutil
import os
import tqdm

dpath = '/DATA/BIO/GDC/downloaded/*/*.svs'
svs_files = glob.glob(dpath)

destination_dir = '/DATA/BIO/GDC/liver/slides'
os.makedirs(destination_dir)

for svs_file in tqdm.tqdm(svs_files):
    base = os.path.basename(svs_file)
    svs_copy = os.path.join(destination_dir, base)
    shutil.copyfile(svs_file, svs_copy)

print("DONE")