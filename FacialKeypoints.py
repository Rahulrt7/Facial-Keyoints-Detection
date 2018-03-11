
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

import numpy as np
import os

from PIL import Image


# In[2]:


VGG_MEAN = [123.68, 116.78, 103.94]


# In[3]:


x_train_path = './processed_data/x_train.npy'
y_train_path = './processed_data/y_train.npy'
x_val_path = './processed_data/x_val.npy'
y_val_path = './processed_data/y_val.npy'
model_path = 'vgg_16.ckpt'
batch_size = 32
num_workers = 4
num_epochs1 = 1500
const_learning_rate1 = 1e-4
dropout_keep_prob = 0.5
weight_decay = 5e-4


# In[4]:


# load training and validation data
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_val = np.load(x_val_path)
y_val = np.load(y_val_path)

total_images = x_train.shape[0]


# In[5]:


def check_accuracy(sess, error, is_training,
                  dataset_init_op):
    """
    Check the accuracy of the model on either train 
    or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    err = []
    while True:
        try:
            mse, op = sess.run(error, {is_training: False})
            err.append(float(mse))
        except tf.errors.OutOfRangeError:
            break
    
    return sum(err)/float(len(err))

def save_checkpoint(saver, init_err, val_err, sess, epoch, name):
    if val_err < init_err:
        init_err = val_err
        save_path = './checks/' + name
        saver.save(sess, 
                    save_path,
                    global_step=epoch,
                    write_meta_graph=True)
    return init_err    

def get_arrays(fold=None):
    if fold == "train":
        return x_train, y_train
    else:
        return x_val, y_val


# In[6]:


def main():
    x_train, y_train = get_arrays(fold="train")
    x_val, y_val = get_arrays(fold="val")
    
    num_classes = 30
    
    graph = tf.Graph()
    with graph.as_default():
        # Standard preprocessing for VGG on ImageNet
        def _parse_function(image, keypoints):
            image = tf.reshape(image, [96, 96, 1])
            
            rgb_image = tf.image.grayscale_to_rgb(image)
            image = tf.cast(rgb_image, tf.float32)
            
            # changing resolution according from 48x48 to 256x256
            resized_image = tf.image.resize_images(image, tf.constant([256, 256]))
            
            return resized_image, keypoints
        
        # Preprocessing for training
        def training_preprocess(image, keypoints):
            # Random Crop
            crop_image = tf.random_crop(image, [224, 224, 3])
            flip_image = tf.image.random_flip_left_right(crop_image)
            
            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = flip_image - means
            
            return centered_image, keypoints
        
        # Preprocessing for validation
        def val_preprocess(image, keypoints):
            # Central Crop
            crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
            
            means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
            centered_image = crop_image - means
            
            return centered_image, keypoints
        
        # DATASET CREATION using tf.contrib.data.Dataset
        
        # Training dataset
        x_train = tf.constant(x_train)
        y_train = tf.constant(y_train)
        
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.map(_parse_function,
                        num_threads=num_workers,
                        output_buffer_size=batch_size)
        train_dataset = train_dataset.map(training_preprocess,
                        num_threads=num_workers, 
                        output_buffer_size=batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000) # don't forget to shuffle
        batched_train_dataset = train_dataset.batch(batch_size)
        
        # Validation dataset
        x_val = tf.constant(x_val)
        y_val = tf.constant(y_val)
        
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.map(_parse_function,
                        num_threads=num_workers, 
                        output_buffer_size=batch_size)
        val_dataset = val_dataset.map(val_preprocess, 
                        num_threads=num_workers, 
                        output_buffer_size=batch_size)
        batched_val_dataset = val_dataset.batch(batch_size)
        
        # Now define the Iterator that can operate on
        # any of the datasets
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.
        
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                          batched_train_dataset.output_shapes)
        
        images, keypoints = iterator.get_next()
        
        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)
        
        # Create a placeholder to indicate if we are in training or validation phase
        is_training = tf.placeholder(tf.bool)
        
        # Get the pretrained VGG16 ready
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
            predictions, _ = vgg.vgg_16(images, num_classes=num_classes, 
                                   is_training=is_training, 
                                  dropout_keep_prob=dropout_keep_prob)
            
        # specify model checkpoint path
        assert(os.path.isfile(model_path))
        
        # Restore only convolutional layers and not fully connected ones
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(exclude=['vgg_16/fc6',
                                                                                    'vgg_16/fc7',
                                                                                      'vgg_16/fc8'])
        
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(model_path, variables_to_restore)
        
        # Initialization operation for all three fully connected layers
        fc6_variables = tf.contrib.framework.get_variables('vgg_16/fc6')
        fc6_init = tf.variables_initializer(fc6_variables)
        fc7_variables = tf.contrib.framework.get_variables('vgg_16/fc7')
        fc7_init = tf.variables_initializer(fc7_variables)
        fc8_variables = tf.contrib.framework.get_variables('vgg_16/fc8')
        fc8_init = tf.variables_initializer(fc8_variables)
        
        # Define loss function 
        tf.losses.mean_squared_error(labels=keypoints,
                                    predictions=predictions)
        loss = tf.losses.get_total_loss()
        
        # For tensorboard
        tf.summary.scalar('loss', loss)
        
        # We will train only for last three layers
        var_list = []
        var_list.append(fc6_variables)
        var_list.append(fc7_variables)
        var_list.append(fc8_variables)
        
        # training op to train complete network
        # with step decay for learning rate
        global_step = tf.Variable(0, trainable=False)
        boundaries = [500, 1000, 1500]
        values = [0.0001, 0.00005, 0.00001, 0.000005]
        learning_rate = tf.train.piecewise_constant(global_step,boundaries,values)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, var_list=var_list) 
    
        # Evaluation Metrics
        error = tf.metrics.mean_squared_error(labels=keypoints,
                                             predictions=predictions)
        
        # Save built checkpoint
        saver = tf.train.Saver(max_to_keep=3)
        
        # Variables Initializer
        global_init = tf.global_variables_initializer()
        local_init = tf.local_variables_initializer()
        
        summary_op = tf.summary.merge_all()
        
        tf.get_default_graph().finalize()
        
    # Graph has been built, time to throw some computation
    with tf.Session(graph=graph) as sess:
        init_fn(sess) # load the pretrained weights
        sess.run(fc6_init) # initialize the new fc6 layer
        sess.run(fc7_init)
        sess.run(fc8_init)
        
        sess.run(global_init)
        sess.run(local_init)

        init_mse = 1000000.0 # take a big number to compare error
        
        writer = tf.summary.FileWriter("./logs", graph=None)
        
        # Update only last three layers
        for epoch in range(num_epochs1):
            # Track loss
            loss_history = []
            
            # number of batches in one epoch
            batch_count = int(total_images/batch_size)
            
            # Run an epoch over the training data
            print("Starting epoch %d / %d" % (epoch + 1,
                                             num_epochs1))
            # Initialize the iterator with training set
            sess.run(train_init_op)
            
            while True:
                try:
                    _, summary = sess.run([train_op, summary_op],
                                 {is_training: True})
                    loss_history.append(sess.run(loss, 
                                                 {is_training: True}))
                except tf.errors.OutOfRangeError:
                    break
            
            # print average loss per epoch
            epoch_loss = sum(loss_history) / float(len(loss_history))
            print("Loss: %f" %  epoch_loss, end="")
            
            # Check accuracy on the train and val sets every epoch
            #train_mse = check_accuracy(sess, error, is_training, train_init_op)
            val_mse = check_accuracy(sess, error, is_training, val_init_op)
            
            #print("Train MSE: %f" % train_mse)
            print("    Validation MSE: %f" % val_mse)
            
            # Call the function to save the checkpoint
            init_mse = save_checkpoint(saver, init_mse, val_mse, sess,
                           epoch, 'my_model1')
            #print("Checkpoint Saved")
            
            # tensorboard logging
            writer.add_summary(summary, epoch)
            #print("Tensorboard logging complete")
        
            


# In[7]:


if __name__ == '__main__':
    main()

