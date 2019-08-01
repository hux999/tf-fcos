import time

from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
import cv2

from model.fcos_detector import build_fcos_detector
from model.fcos_loss import fcos_loss
from dataset_tools import detection_dataset 

flags.DEFINE_string('dataset', './data/coco_tf_records/*', 'path to dataset')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_integer('batch_size', 8, 'batch size')


def vis_dataset(image, bbox):
    image = image.numpy()[0, :, :, :]
    bbox = bbox.numpy()[0, :, :]
    image = image.astype(np.uint8)
    for i in range(bbox.shape[0]):
        bb = bbox[i, :]
        cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0,255,0), 2)
    cv2.imshow("image", image)
    cv2.waitKey()


def build_dataset():
    train_dataset = detection_dataset.load_tfrecord_dataset(FLAGS.dataset, FLAGS.classes)
    train_dataset = train_dataset.map(
        lambda x, y: detection_dataset.resize_and_pad(x, y, 800, 1024)
    )
    train_dataset = train_dataset.map(detection_dataset.random_horizon_flip)
    train_dataset = train_dataset.shuffle(buffer_size=64).batch(8).prefetch(16)
    return train_dataset


'''
def main(_argv):
    model = build_fcos_detector(80)
    train_dataset = detection_dataset.load_tfrecord_dataset(FLAGS.dataset, FLAGS.classes)
    train_dataset = train_dataset.map(
        lambda x, y: detection_dataset.resize_and_pad(x, y, 800, 1344)
    )
    train_dataset = train_dataset.map(detection_dataset.random_horizon_flip)
    train_dataset = train_dataset.batch(2)
    for i, (image, gt_boxes) in enumerate(train_dataset):
        #vis_dataset(image, gt_boxes)
        logits, bbox_reg, centerness = model(image)
        losses = fcos_loss(logits, bbox_reg,  centerness, gt_boxes)
        print("cls_loss", losses[0])
        print("centerness_loss", losses[1])
        print("reg_loss", losses[2])
        if i > 10:
            break
'''


def train_():
    mirrored_strategy = tf.distribute.MirroredStrategy()

    num_epochs = 20
    save_iter = 1
    train_log_dir = "./output/tensorboard"
    checkpoint_dir = "./output/model/"
    
    with mirrored_strategy.scope():
        # build dataset
        model = build_fcos_detector(80)
        optimizer = tf.keras.optimizers.SGD(0.001, momentum=0.9)

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        train_loss_mean = tf.keras.metrics.Mean(name='loss')
        cls_loss_mean = tf.keras.metrics.Mean(name='cls_loss')
        reg_loss_mean = tf.keras.metrics.Mean(name='reg_loss')
        centerness_loss_mean = tf.keras.metrics.Mean(name='centerness_loss')
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        train_dataset = build_dataset()
        train_dataset = mirrored_strategy.experimental_distribute_dataset(train_dataset)

        def step_fn(inputs):
            image_data, target = inputs

            with tf.GradientTape() as tape:
                logits, bbox_reg, centerness = model(image_data, training=True)
                cls_loss, centerness_loss, reg_loss = \
                    fcos_loss(logits, bbox_reg, centerness, target)
                loss = cls_loss + centerness_loss + reg_loss
        
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))

            train_loss_mean.update_state(loss)
            cls_loss_mean.update_state(cls_loss)
            reg_loss_mean.update_state(reg_loss)
            centerness_loss_mean.update_state(centerness_loss)
            
            return loss
        
        @tf.function
        def train_step(dist_inputs):
            per_example_losses = mirrored_strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
            #mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
            #return mean_loss

        i_batch = 0
        for i_epoch in range(num_epochs):
            for batch_data in train_dataset:
                start = time.time()
                train_step(batch_data)
                end = time.time()

                if i_batch % 1 == 0:
                    with train_summary_writer.as_default():
                        tf.summary.scalar("loss", train_loss_mean.result(), step=i_batch)
                        tf.summary.scalar("cls_loss", cls_loss_mean.result(), step=i_batch)
                        tf.summary.scalar("reg_loss", reg_loss_mean.result(), step=i_batch)
                        tf.summary.scalar("centerness_loss", centerness_loss_mean.result(), step=i_batch)
                    print("[%d/%d] loss: %.5f" % (i_batch, i_epoch, train_loss_mean.result()))
                    print("\tcls_loss: %.5f" % cls_loss_mean.result())
                    print("\treg_loss: %.5f" % reg_loss_mean.result())
                    print("\tcenterness_loss: %.5f" % centerness_loss_mean.result())
                    print("%.1fms/iter" % ((end-start)*1000))
                i_batch += 1

            if (i_epoch+1) % save_iter == 0:
                checkpoint.save(checkpoint_dir)


def main(_argv):

    # train
    train_()


if __name__ == '__main__':
    tf.debugging.set_log_device_placement(True)
    app.run(main)
