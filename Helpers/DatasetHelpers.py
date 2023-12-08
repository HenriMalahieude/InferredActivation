import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers, models

#The copy and pasting I'm doing now is pretty criminal, which is why I'm fixing it
def prepare_dataset(ds, img_shape, augment_factor=0.0, concatenate_augment=False):
	resize_and_rescale = models.Sequential([
	layers.Resizing(*img_shape),
	layers.Rescaling(1./255),
	])

	data_augmentation = models.Sequential([
		layers.RandomFlip(),
		layers.RandomContrast(augment_factor),
		layers.RandomBrightness(augment_factor),
		layers.RandomRotation(augment_factor),
		resize_and_rescale
	])

	dsn = ds.map((lambda x, y: (resize_and_rescale(x, training=True), y)), num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

	if augment_factor > 0:
		augmenter = (lambda x, y: (data_augmentation(x, training=True), y))
		if concatenate_augment:
			dsn = dsn.concatenate(ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False))
		else:
			dsn = ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

	print("\tDataset Prepared")
	return dsn.prefetch(buffer_size=tf.data.AUTOTUNE)

def load_cifar10(batch_size=8):
	training, validation = tf.keras.datasets.cifar10.load_data()
	train_ds = tf.data.Dataset.from_tensor_slices(training).batch(batch_size)
	val_ds = tf.data.Dataset.from_tensor_slices(validation).batch(batch_size)
	return train_ds, val_ds

def report_dataset_size(name, dataset, batch_sz):
	print("\t\t{} Images Loaded for {}".format(dataset.cardinality() * batch_sz, name))