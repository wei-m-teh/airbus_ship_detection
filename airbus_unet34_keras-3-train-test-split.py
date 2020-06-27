BATCH_SIZE = 4
EDGE_CROP = 16
NB_EPOCHS = 20
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# downsampling inside the network
NET_SCALING = None
# downsampling in preprocessing
IMG_SCALING = (1, 1)
# number of validation images to use
VALID_IMG_COUNT= 4
VALIDATION_COUNT_PCT= 0.001
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 500
AUGMENT_BRIGHTNESS = False

import argparse
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import os
import gc; gc.enable() # memory is tight
import tensorflow as tf
from skimage.morphology import label

def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction

def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)

def sample_ships_for_validation(in_df, total, valid_pct):
    group_cnt = in_df['ImageId'].count()
    frac = int(total * valid_pct * group_cnt // total)
    print("group cnt: {}, fraction: {}".format(group_cnt, frac))
    return in_df.sample(frac, replace=False)

def make_image_gen(in_df, batch_size = BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []

if __name__ =='__main__':
  parser = argparse.ArgumentParser()

  # hyperparameters sent by the client are passed as command-line arguments to the script.
  parser.add_argument('--ship-dir', type=str, default='/mnt/fsx/airbus_data/deep_learning_v2')
  parser.add_argument('--train-dir', type=str, required=True)
  parser.add_argument('--mask-csv', type=str, required=True)
  parser.add_argument('--output-train-csv', type=str, required=True)
  parser.add_argument('--output-valid-csv', type=str, required=True)

  args, _ = parser.parse_known_args()

  ship_dir = args.ship_dir
  train_image_dir = os.path.join(ship_dir, args.train_dir)
  masks = pd.read_csv(os.path.join(ship_dir, args.mask_csv))

  masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
  unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
  unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
  unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])

  masks.drop(['ships'], axis=1, inplace=True)
  #unique_img_ids.sample(5)

  train_df = pd.merge(masks, unique_img_ids)
  train_df['grouped_ship_count'] = train_df['ships'].map(lambda x: (x + 1) // 2).clip(0, 7)
  grouped_train_df = train_df.groupby('grouped_ship_count')

  total = train_df['ImageId'].count()
  valid_df = grouped_train_df.apply(sample_ships_for_validation, total, VALIDATION_COUNT_PCT)
  all_train_df = train_df[~train_df['ImageId'].isin(valid_df['ImageId'])]
 
  print("training samples: {}, validation sample size: {}".format(all_train_df['ImageId'].count(), valid_df['ImageId'].count()))
  training_samples = all_train_df['ImageId'].count()
  validation_samples = valid_df['ImageId'].count()
 
  valid_gen = make_image_gen(valid_df, validation_samples) # pull all validation samples
  valid_x, valid_y = next(valid_gen)
  print('x', valid_x.shape, valid_x.min(), valid_x.max())
  print('y', valid_y.shape, valid_y.min(), valid_y.max())

  # write to CSV 
  masks[masks['ImageId'].isin(all_train_df['ImageId'])].to_csv('{}/{}'.format(ship_dir, args.output_train_csv), index=False)
  masks[masks['ImageId'].isin(valid_df['ImageId'])].to_csv('{}/{}'.format(ship_dir, args.output_valid_csv), index=False)
