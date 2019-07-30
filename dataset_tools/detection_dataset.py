import tensorflow as tf


IMAGE_FEATURE_MAP = {
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    'image/object/view': tf.io.VarLenFeature(tf.string),
}

def flip_box(image, bbox):
    image_width = tf.cast(tf.shape(image)[1], tf.float32)
    bbox = tf.stack([
        image_width-bbox[:, 2],
        bbox[:, 1],
        image_width-bbox[:, 0],
        bbox[:, 3],
        bbox[:, 4],
    ], axis=-1)
    return bbox

def random_horizon_flip(image, bbox):
    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
    image, bbox = tf.cond(
        choice < 0.5,
        lambda: (tf.image.flip_left_right(image), flip_box(image, bbox)),
        lambda: (image, bbox)
    )
    return image, bbox


def resize_and_pad(image, bbox, min_size=800, max_size=1333):
    init_height = tf.cast(tf.shape(image)[0], tf.float32)
    init_width = tf.cast(tf.shape(image)[1], tf.float32)
    im_max_size = tf.maximum(init_height, init_width)
    im_min_size = tf.minimum(init_height, init_width)

    scale = tf.minimum(max_size/im_max_size, min_size/im_min_size)
    new_height = tf.cast(init_height*scale+0.5, tf.int32)
    new_width = tf.cast(init_width*scale+0.5, tf.int32)

    image = tf.image.resize(
        image, 
        [new_height, new_width],
        method=tf.image.ResizeMethod.BILINEAR,
        preserve_aspect_ratio=False,
        antialias=False,
    )
    bbox = tf.concat([scale*bbox[:, :4], tf.expand_dims(bbox[:,4], 1)], axis=1)

    paddings = [[0, max_size-new_height], [0, max_size-new_width], [0, 0]]
    image = tf.pad(image, paddings, mode="CONSTANT", constant_values=-1)

    return image, bbox


def parse_tfrecord(tfrecord, class_table):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    image = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    image = tf.cast(image, tf.float32)
    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    bbox = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin'])*image_width,
                        tf.sparse.to_dense(x['image/object/bbox/ymin']*image_height),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']*image_width),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']*image_height),
                        labels], axis=1)

    paddings = [[0, 100 - tf.shape(bbox)[0]], [0, 0]]
    bbox = tf.pad(bbox, paddings)

    return image, bbox


def load_tfrecord_dataset(file_pattern, class_file):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table))