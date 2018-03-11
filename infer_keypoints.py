import tensorflow as tf
import time
import pandas as pd
import numpy as np

dir(tf.contrib)

VGG_MEAN = [123.68, 116.78, 103.94]

"""
# read test data from csv fiel
test_df = pd.read_csv("./data/test.csv", sep=",")

# convert image pixels(str) to numpy(float32) arrays
test_images = test_df["Image"]
flat_image_size = len(test_images[0].split())
x_test = np.zeros((test_df.shape[0],flat_image_size), dtype=np.float32)
index = 0
for image in test_images:
    pixels_str = image.split()
    # Using map function to convert all string values to float32
    pixels = list(map(np.float32, pixels_str))
    x_test[index] = np.asarray(pixels)
    index += 1
"""

"""
Directly load test data from .npy file rather than 
first reading from from csv file and then converting to 
numpy array (for fast loading)
"""
x_test = np.load("./processed_data/x_test.npy")
print(x_test.shape)

num_classes = 30

def get_keypoints(detection_graph, sess, image_np):                 
    image = tf.reshape(image_np, [96, 96, 1])        
    rgb_image = tf.image.grayscale_to_rgb(image)
    image = tf.cast(rgb_image, tf.float32)
    resized_image = tf.image.resize_images(image, tf.constant([256, 256]))
    crop_image = tf.image.resize_image_with_crop_or_pad(resized_image, 224, 224)
    means = tf.reshape(tf.constant(VGG_MEAN), [1, 1, 3])
    centered_image = crop_image - means
    batched_image = tf.reshape(centered_image, [1, 224, 224, 3])
    np_image = sess.run(batched_image)
    
    image_tensor = detection_graph.get_tensor_by_name("IteratorGetNext:0")
    is_training_tensor = detection_graph.get_tensor_by_name("Placeholder:0")
    # fetch and run prediction tensor_op
    prediction_op = detection_graph.get_tensor_by_name("vgg_16/fc8/squeezed:0")
            
    tic = time.time()
    pred = sess.run(prediction_op, 
                    feed_dict={image_tensor: np_image, 
                               is_training_tensor: False})
    tac = time.time()
            
    print("Time Taken :: %f" % (tac-tic))
    return pred[0] #as pred is list of list

def display_points(points, image_np):
    # convert keypoint values(30) to facial keypoints coordinates(15)
    point_xy_tuple = []
    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i+1]
        point_xy_tuple.append((x, y))
    print("15 Facial keypoints coordinates::")
    print(point_xy_tuple)

if __name__ == "__main__":
    
    """Index of the image to be teseted, Change the value of index to test on different image"""
    # index value should be between 0-1783
    index = 1
    
    # Initilaize variable names
    MODEL_FOLDER = './checks'
    
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_META = MODEL_FOLDER + '/my_model1-699.meta'    
    
    #Load a frozen Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        new_saver = tf.train.import_meta_graph(PATH_TO_META)
        with tf.Session(graph=detection_graph) as sess:
            ckpt = MODEL_FOLDER + '/my_model1-699'
            new_saver.restore(sess, ckpt)
            # get the keypoints and print them        
            points = get_keypoints(detection_graph, sess, x_test[index])
            
            display_points(points, x_test[index])
                        
                        
        
