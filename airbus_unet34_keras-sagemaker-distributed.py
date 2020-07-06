import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
import horovod.tensorflow.keras as hvd
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import segmentation_models as sm
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from skimage.morphology import label
import argparse
import gc; gc.enable()
from tensorflow.keras.callbacks import Callback

IMG_SCALING = (1, 1)

# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 500

class ModelPerformanceMetricsCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("For epoch {}, train_loss={}, train_dice_coef={}, train_binary_accuracy={}, train_true_positive_rate={}, val_loss={}, val_dice_coef={}, val_bin_accuracy={}, val_true_positive_rate={}".format(
            epoch, logs['loss'], logs['dice_coef'], logs['binary_accuracy'], logs['true_positive_rate'], logs['val_loss'], logs['val_dice_coef'], logs['val_binary_accuracy'],
            logs['val_true_positive_rate']))

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

def make_image_gen(in_df, batch_size = 4):
    out_rgb = []
    out_mask = []
    for _, row in in_df.iterrows():
        c_img_id = row['ImageId']
        masks = row['EncodedPixels']
        rgb_path = os.path.join(train_image_dir, c_img_id)
        c_img = imread(rgb_path)
        c_mask = masks_as_image([masks])
        if IMG_SCALING is not None:
            c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
            c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
        out_rgb += [c_img]
        out_mask += [c_mask]
        if len(out_rgb)>=batch_size:
            yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
            out_rgb, out_mask=[], []

def dice_loss(y_true, y_pred):
     smooth = 1.
     y_true_f = K.flatten(y_true)
     y_pred_f = K.flatten(y_pred)
     intersection = tf.dtypes.cast(y_true_f, tf.float32) * y_pred_f
     score = (2. * K.sum(intersection) + smooth) / (K.sum(tf.dtypes.cast(y_true_f, tf.float32)) + K.sum(y_pred_f) + smooth)
     return 1. - score

def bce_dice_loss(y_true, y_pred):
  return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def bce_logdice_loss(y_true, y_pred):
  return binary_crossentropy(y_true, y_pred) - K.log(1. - dice_loss(y_true, y_pred))

def dice_coef(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_p_bce(in_gt, in_pred):
  return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def true_positive_rate(y_true, y_pred):
  return K.sum(K.flatten(y_true)*K.flatten(K.round(y_pred)))/K.sum(y_true)

if __name__ =='__main__':

   parser = argparse.ArgumentParser()

   # hyperparameters sent by the client are passed as command-line arguments to the script.
   parser.add_argument('--epochs', type=int, default=2)
   parser.add_argument('--batch-size', type=int, default=8)
   parser.add_argument('--learning-rate', type=float, default=1e-4)

   # input data and model directories
   parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
   parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING']) # default training directory being the Fsx for Lustre

   args, _ = parser.parse_known_args()

   hvd.init()
   gpus = tf.config.experimental.list_physical_devices('GPU')
   for gpu in gpus:
     tf.config.experimental.set_memory_growth(gpu, True)
   if gpus:
     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

   train_image_dir = os.path.join(args.train, 'train')

   train_masks = pd.read_csv(os.path.join(args.train, 'train_ship_segmentations_train.csv'))

   train_gen = make_image_gen(train_masks, args.batch_size)
   train_x, train_y = next(train_gen)

   valid_masks = pd.read_csv(os.path.join(args.train, 'train_ship_segmentations_valid.csv'))
   print(valid_masks.shape[0], 'masks found')

   valid_gen = make_image_gen(valid_masks, args.batch_size)
   valid_x, valid_y = next(valid_gen)

   # Contruct a model using UNET with RESTNET 34
   sm.set_framework('tf.keras')
   K.set_image_data_format('channels_last')
   model = sm.Unet('resnet34', classes=1, input_shape=(768,768,3), encoder_weights='imagenet')
   model.summary()

   lr = args.learning_rate
   scaled_lr = lr * hvd.size()
   weight_path="{}/{}_weights.best.hdf5".format(args.model_dir,'seg_model')

   checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                             save_best_only=True, mode='max', save_weights_only = True)

   reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                    patience=3,
                                    verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
   early = EarlyStopping(monitor="val_dice_coef",
                      mode="max",
                      patience=30)

   hvdBroadcast = hvd.callbacks.BroadcastGlobalVariablesCallback(0)
   hvdMetricsAvg = hvd.callbacks.MetricAverageCallback()
   # hvdLRWarmup = hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, initial_lr=scaled_lr, verbose=1)
    
   callbacks_list = [hvdBroadcast, hvdMetricsAvg]

   # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
   if hvd.rank() == 0:
     model_perf_metrics = ModelPerformanceMetricsCallback()
     callbacks_list.append(checkpoint)
     callbacks_list.append(model_perf_metrics)

   #opt = Adam(scaled_lr, decay=1e-6)
   opt = Adam(scaled_lr)
   opt = hvd.DistributedOptimizer(opt)
   model.compile(optimizer=opt, loss=bce_logdice_loss, metrics=[dice_coef, 'binary_accuracy', true_positive_rate], experimental_run_tf_function=False)

   train_steps = (train_masks.shape[0] // args.batch_size) // hvd.size()
   valid_steps = (valid_masks.shape[0] // args.batch_size) // hvd.size()
   step_count = min(MAX_TRAIN_STEPS, train_steps)
   loss_history = [model.fit(train_gen,
                           # batch_size=args.batch_size,
                          epochs=args.epochs,
                          steps_per_epoch=step_count,
                          validation_data=valid_gen,
                          validation_steps=valid_steps,
                          callbacks=callbacks_list,
                          verbose=1 if hvd.rank() == 0 else 0)]
