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
        lambda x, y: detection_dataset.resize_and_pad(x, y, 800, 1344)
    )
    train_dataset = train_dataset.map(detection_dataset.random_horizon_flip)
    train_dataset = train_dataset.batch(2)
    return train_dataset


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
        break


def train_(model, train_dataset):
    num_epochs = 20
    save_iter = 1
    train_log_dir = "./output/tensorboard"
    checkpoint_dir = "./output/model/"

    optimizer = tf.keras.optimizers.Adam(0.0001)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    train_loss_mean = tf.keras.metrics.Mean(name='loss')
    cls_loss_mean = tf.keras.metrics.Mean(name='cls_loss')
    reg_loss_mean = tf.keras.metrics.Mean(name='reg_loss')
    centerness_loss_mean = tf.keras.metrics.Mean(name='centerness_loss')

    #test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)         

    def train_step(inputs):
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

    i_batch = 0
    for i_epoch in range(num_epochs):
        for batch_data in train_dataset:
            train_step(batch_data)

            if i_batch % 1 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", train_loss_mean.result(), step=i_batch)
                print("[%d/%d] loss: %.5f, cls_loss: %.5f" %
                    (i_batch, i_epoch, train_loss_mean.result(), cls_loss_mean.result()))
            i_batch += 1

        if (i_epoch+1) % save_iter == 0:
            checkpoint.save(checkpoint_dir)
        
        train_loss.reset_states()
        train_accuracy.reset_states()

'''
def main(_argv):
    # build dataset
    train_dataset = build_dataset()
    model = build_fcos_detector(80)

    # train
    train_(model, train_dataset)
'''


if __name__ == '__main__':
    app.run(main)