import argparse
import numpy as np
import tensorflow as tf
from utils import *

GRAPH = 'retrained_graph.pb'
with open('labels.txt', 'r') as f:
    LABELS = f.read().split()

def compute_brute_force_classification(image_path, nH=8, nW=8):
    '''
    This function returns the probabilities of each window.
    Inputs:
        image_path: path to the image to be analysed
        nH: number of windows in the vertical direction
        nW: number of windows in the horizontal direction
    Outputs:
        window_predictions: a (nH, nW, 3) np.array. 
                            The last dim (size 3) is the probabilities 
                            of each label (cat, dog, neg)
    HINT: classify_image  (from utils.py) will be useful here.
    HINT: You will need to define a tensorflow session. You can do this by running:
          with tf.Session() as sess: 
    '''

    # H x W x 3 numpy array (3 for each RGB color channel)
    raw_image = decode_jpeg(image_path)
    ######### Your code starts here #########
    height, width, _ = raw_image.shape
    # Normalizing
    d_height = np.floor(float(height)/float(nH))
    # Normalizing
    d_width = np.floor(float(width)/float(nW))

    #I tried between 1 and 100 and 45 seemed to give the best results
    padding_amount = 45

    after_padding = np.pad(raw_image, ((padding_amount, padding_amount), (padding_amount, padding_amount), (0,0)),mode = 'constant')
    my_prediction = np.empty((nH,nW,3))

    with tf.Session() as sess:
    	for i in range(nH):
    		for j in range(nW):
    			window = after_padding[int(i*d_height):int((i+1)*d_height+2*padding_amount), int(j*d_width):int((j+1)*d_width + 2 * padding_amount), :]
    			window_prediction = classify_image(window,sess)
    			my_prediction[i,j,:] = window_prediction
    '''
    with tf.Session() as sess:
        window = raw_image    # the "window" here is the whole image
        window_prediction = classify_image(window, sess)

    # setting each window prediction to be the prediction for the whole image (just for something to plot)
    # do not turn this code in and claim that "your window padding is infinite"
    window_predictions = np.array([[window_prediction for c in range(nW)] for r in range(nH)])
    '''

    ######### Your code ends here #########

    return my_prediction

def compute_convolutional_KxK_classification(image):
    graph = tf.get_default_graph()
    classification_input_tensor =  graph.get_tensor_by_name('bottleneck_input/BottleneckInputPlaceholder:0')
    classification_output_tensor = graph.get_tensor_by_name('final_result:0')
    convolution_ouput_tensor = graph.get_tensor_by_name('bottleneck_grid:0')    # 'mixed_10/join:0' in Inception-v3
    K = convolution_ouput_tensor.shape[0]

    with tf.Session() as sess:
        convolution_output = run_with_image_input(convolution_ouput_tensor, image, sess)
        predictionsKxK = sess.run(classification_output_tensor,
                                  {classification_input_tensor: np.reshape(convolution_output, [K*K, -1])})
        return np.reshape(predictionsKxK, [K,K,-1])

def compute_and_plot_saliency(image):
    '''
    This function computes and plots the saliency plot.
    Input: image to be analysed
    Output: None

    You need to compute the matrix M detailed in section 3.1 in
    K. Simonyan, A. Vedaldi, and A. Zisserman, 
    "Deep inside convolutional networks: Visualising imageclassification models and saliency maps," 
    2013, Available at https://arxiv.org/abs/1312.6034.
    '''
    graph = tf.get_default_graph()
    logits_tensor = graph.get_tensor_by_name('final_training_ops/Wx_plus_b/logits:0')

    with tf.Session() as sess:
        logits = np.squeeze(run_with_image_input(logits_tensor, image, sess))
        top_class = np.argmax(logits)
        w_ijc = gradient_of_class_score_with_respect_to_input_image(image, top_class, sess)    # defined in utils.py
    ######### Your code starts here #########
    
    #M = np.zeros(raw_gradients.shape[0:2])  # something of the right shape to plot
    M = np.amax(np.abs(w_ijc), axis =2)
    ######### Your code ends here #########

    plt.subplot(2,1,1)
    plt.imshow(M)
    plt.title('Saliency with respect to predicted class %s' % LABELS[top_class])
    plt.subplot(2,1,2)
    plt.imshow(decode_jpeg(image))
    plt.show()

def plot_classification(image_path, classification_array):
    nH, nW, _ = classification_array.shape
    image_data = decode_jpeg(image_path)
    aspect_ratio = float(image_data.shape[0]) / image_data.shape[1]
    plt.figure(figsize=(8, 8*aspect_ratio))
    p1 = plt.subplot(2,2,1)
    plt.imshow(classification_array[:,:,0], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[0])
    p1.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    p2 = plt.subplot(2,2,2)
    plt.imshow(classification_array[:,:,1], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[1])
    p2.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    p2 = plt.subplot(2,2,3)
    plt.imshow(classification_array[:,:,2], interpolation='none', cmap='jet')
    plt.title('%s probability' % LABELS[2])
    p2.set_aspect(aspect_ratio*nW/nH)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(image_data)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str)
    parser.add_argument('--scheme', type=str)
    FLAGS, _ = parser.parse_known_args()

    load_graph(GRAPH)
    if FLAGS.scheme == 'brute':
        plot_classification(FLAGS.image, compute_brute_force_classification(FLAGS.image, 8, 8))
    elif FLAGS.scheme == 'conv':
        plot_classification(FLAGS.image, compute_convolutional_KxK_classification(FLAGS.image))
    elif FLAGS.scheme == 'saliency':
        compute_and_plot_saliency(FLAGS.image)
    else:
        print 'Unrecognized scheme:', FLAGS.scheme