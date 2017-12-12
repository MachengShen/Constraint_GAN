import os
import numpy as np
import prettytensor as pt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from deconv import deconv2d
import IPython.display
import math
import tqdm # making loops prettier
import h5py # for reading our dataset
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed

def create_image(im):
        return np.reshape(im,(dim1,dim2,dim3))


def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        idxs = np.arange(0, len(faces))
        np.random.shuffle(idxs)
        for batch_idx in range(0, len(faces), batch_size*num_cpus):
            cur_idxs = idxs[batch_idx:batch_idx+batch_size*num_cpus]
            images_batch = faces[cur_idxs]
            #images_batch = images_batch.astype("float32")
            labels_batch = labels[cur_idxs]
            yield images_batch, labels_batch



def encoder(X):
    '''Create encoder network.
    Args:
        x: a batch of flattened images [batch_size, 28*28]
    Returns:
        A tensor that expresses the encoder network
            # The transformation is parametrized and can be learned.
            # returns network output, mean, setd
    '''
    lay_end = (pt.wrap(X).
            reshape([batch_size, dim1, dim2, dim3]).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, stride=2).
            conv2d(5, 256, stride=2).
            flatten())
    z_mean = lay_end.fully_connected(hidden_size, activation_fn=None)
    z_log_sigma_sq = lay_end.fully_connected(hidden_size, activation_fn=None)
    return z_mean, z_log_sigma_sq


def generator(Z):
    '''Create generator network.
        If input tensor is provided then decodes it, otherwise samples from
        a sampled vector.
    Args:
        x: a batch of vectors to decode
    Returns:
        A tensor that expresses the generator network
    '''
    return (pt.wrap(Z).
            fully_connected(8*8*256).reshape([batch_size, 8, 8, 256]). #(128, 4 4, 256)
            deconv2d(5, 256, stride=2).
            deconv2d(5, 128, stride=2).
            deconv2d(5, 32, stride=2).
            deconv2d(1, dim3, stride=1, activation_fn=tf.sigmoid).
            flatten()
           )



def discriminator(D_I):
    ''' A encodes
    Create a network that discriminates between images from a dataset and
    generated ones.
    Args:
        input: a batch of real images [batch, height, width, channels]
    Returns:
        A tensor that represents the network
    '''
    descrim_conv =  (pt.wrap(D_I). # This is what we're descriminating
            reshape([batch_size, dim1, dim2, dim3]).
            conv2d(5, 32, stride=1).
            conv2d(5, 128, stride=2).
            conv2d(5, 256, stride=2).
            conv2d(5, 256, stride=2).
            flatten()
            )
    lth_layer= descrim_conv.fully_connected(1024, activation_fn=tf.nn.elu)# this is the lth layer
    D =lth_layer.fully_connected(1, activation_fn=tf.nn.sigmoid) # this is the actual discrimination
    return D, lth_layer



def inference(x):
    """
    Run the models. Called inference because it does the same thing as tensorflow's cifar tutorial
    """
    z_p =  tf.random_normal((batch_size, hidden_size), 0, 1) # normal dist for GAN
    eps = tf.random_normal((batch_size, hidden_size), 0, 1) # normal dist for VAE

    with pt.defaults_scope(activation_fn=tf.nn.elu,
                               batch_normalize=True,
                               learned_moments_update_rate=0.0003,
                               variance_epsilon=0.001,
                               scale_after_normalization=True):

        with tf.variable_scope("enc"):
                z_x_mean, z_x_log_sigma_sq = encoder(x) # get z from the input
        with tf.variable_scope("gen"):
            z_x = tf.add(z_x_mean,
                tf.multiply(tf.sqrt(tf.exp(z_x_log_sigma_sq)), eps)) # grab our actual z
            x_tilde = generator(z_x)
        with tf.variable_scope("dis"):
            _, l_x_tilde = discriminator(x_tilde)
        with tf.variable_scope("gen", reuse=True):
            x_p = generator(z_p)
        with tf.variable_scope("dis", reuse=True):
            d_x, l_x = discriminator(x)  # positive examples
        with tf.variable_scope("dis", reuse=True):
            d_x_p, _ = discriminator(x_p)
        return z_x_mean, z_x_log_sigma_sq, z_x, x_tilde, l_x_tilde, x_p, d_x, l_x, d_x_p, z_p

def loss(x, x_tilde, z_x_log_sigma_sq, z_x_mean, d_x, d_x_p, l_x, l_x_tilde, dim1, dim2, dim3):
    """
    Loss functions for SSE, KL divergence, Discrim, Generator, Lth Layer Similarity
    """
    ### We don't actually use SSE (MSE) loss for anything (but maybe pretraining)
    SSE_loss = tf.reduce_mean(tf.square(x - x_tilde)) # This is what a normal VAE uses

    # We clip gradients of KL divergence to prevent NANs
    KL_loss = tf.reduce_sum(-0.5 * tf.reduce_sum(1 + tf.clip_by_value(z_x_log_sigma_sq, -10.0, 10.0)
                                   - tf.square(tf.clip_by_value(z_x_mean, -10.0, 10.0) )
                                   - tf.exp(tf.clip_by_value(z_x_log_sigma_sq, -10.0, 10.0) ), 1))/dim1/dim2/dim3
    # Discriminator Loss
    D_loss = tf.reduce_mean(-1.*(tf.log(tf.clip_by_value(d_x,1e-5,1.0)) +
                                    tf.log(tf.clip_by_value(1.0 - d_x_p,1e-5,1.0))))
    # Generator Loss
    G_loss = tf.reduce_mean(-1.*(tf.log(tf.clip_by_value(d_x_p,1e-5,1.0))))# +
                                    #tf.log(tf.clip_by_value(1.0 - d_x,1e-5,1.0))))
    # Lth Layer Loss - the 'learned similarity measure'
    LL_loss = tf.reduce_sum(tf.square(l_x - l_x_tilde))/dim1/dim2/dim3
    return SSE_loss, KL_loss, D_loss, G_loss, LL_loss

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.


    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def plot_network_output():
    """ Just plots the output of the network, error, reconstructions, etc


    random_x, recon_z, all_d= sess.run((x_p, z_x_mean, d_x_p), {all_input: example_data})
    top_d = np.argsort(np.squeeze(all_d))
    recon_x = sess.run((x_tilde), {z_x: recon_z})
    examples = 8
    random_x = np.squeeze(random_x)
    recon_x = np.squeeze(recon_x)
    random_x = random_x[top_d]

    fig, ax = plt.subplots(nrows=3,ncols=examples, figsize=(18,6))
    for i in xrange(examples):
        ax[(0,i)].imshow(create_image(random_x[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(1,i)].imshow(create_image(recon_x[i]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(2,i)].imshow(create_image(example_data[i + (num_cpus-1)*batch_size]), cmap=plt.cm.gray, interpolation='nearest')
        ax[(0,i)].axis('off')
        ax[(1,i)].axis('off')
        ax[(2,i)].axis('off')
    fig.suptitle('Top: random points in z space | Bottom: inputs | Middle: reconstructions')
    plt.show()
    #fig.savefig(''.join(['imgs/test_',str(epoch).zfill(4),'.png']),dpi=100)
    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,10), linewidth = 4)
    KL_plt, = plt.semilogy((KL_loss_list), linewidth = 4, ls='-', color='r', alpha = .5, label='KL')
    D_plt, = plt.semilogy((D_loss_list),linewidth = 4, ls='-', color='b',alpha = .5, label='D')
    G_plt, = plt.semilogy((G_loss_list),linewidth = 4, ls='-', color='k',alpha = .5, label='G')
    SSE_plt, = plt.semilogy((SSE_loss_list),linewidth = 4,ls='-', color='g',alpha = .5, label='SSE')
    LL_plt, = plt.semilogy((LL_loss_list),linewidth = 4,ls='-', color='m',alpha = .5, label='LL')

    axes = plt.gca()
    leg = plt.legend(handles=[KL_plt, D_plt, G_plt, SSE_plt, LL_plt], fontsize=20)
    leg.get_frame().set_alpha(0.5)
    plt.show()
    """



def sigmoid(x,shift,mult):
    """
    Using this sigmoid to discourage one network overpowering the other
    """
    return 1 / (1 + math.exp(-(x+shift)*mult))

iter_ = data_iterator()


if __name__== "__main__":
    print "Initializing variables."
    dim1 = 64 # first dimension of input data
    dim2 = 64 # second dimension of input data
    dim3 = 3 # third dimension of input data (colors)
    batch_size = 32 # size of batches to use (per GPU)
    hidden_size = 2048 # size of hidden (z) layer to use
    num_examples = 60000 # how many examples are in your training set
    num_epochs = 30 # number of epochs to run
### we can train our different networks  with different learning rates if we want to
    e_learning_rate = 1e-3
    g_learning_rate = 2e-3
    d_learning_rate = 1e-3
    dir_name = 'celeba_constrained_balanced_loss'
    constraint_boundary = -np.float32(np.log(0.5))

    G_loss_list = []
    D_loss_list = []
    SSE_loss_list = []
    KL_loss_list = []
    LL_loss_list = []
    dxp_list = []
    dx_list = []
    gpus = [] # Here I set CUDA to only see one GPU
    os.environ["CUDA_VISIBLE_DEVICES"]=','.join([str(i) for i in gpus])
    num_gpus = len(gpus) # number of GPUs to use
    cpus = [0,1,2,3, 4, 5, 6, 7, 8, 9]
    num_cpus = len(cpus)

    print("Loading data.")
    with h5py.File(''.join(['datasets/faces_dataset_new.h5']), 'r') as hf:
        faces = hf['images'].value
        headers = hf['headers'].value
        labels = hf['label_input'].value

    #with tf.Graph().as_default(), tf.device('/cpu:0'):
    # Create a variable to count number of train calls
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)


    # different optimizers are needed for different learning rates (using the same learning rate seems to work fine though)
    lr_D = tf.placeholder(tf.float32, shape=[])
    lr_G = tf.placeholder(tf.float32, shape=[])
    lr_E = tf.placeholder(tf.float32, shape=[])
    with tf.variable_scope(tf.get_variable_scope(),reuse=None):
        opt_D = tf.train.AdamOptimizer(lr_D, epsilon=1.0)
        opt_G = tf.train.AdamOptimizer(lr_G, epsilon=1.0)
        opt_E = tf.train.AdamOptimizer(lr_E, epsilon=1.0)
    tower_grads_e = []
    tower_grads_g = []
    tower_grads_d = []

    all_input = tf.placeholder(tf.float32, [batch_size*num_cpus, dim1*dim2*dim3])
    KL_param = tf.placeholder(tf.float32)
    LL_param = tf.placeholder(tf.float32)
    G_param = tf.placeholder(tf.float32)

    print("Setting up graph.")
    # Define the network for each GPU
    for i in xrange(num_cpus):
          with tf.device('/cpu:%d' % i):
                with tf.variable_scope('Tower' , reuse=(i>0)):

                #with tf.name_scope('Tower_%d' % (i)) as scope:
                        # grab this portion of the input
                    next_batch = all_input[i*batch_size:(i+1)*batch_size,:]

                    # Construct the model
                    z_x_mean, z_x_log_sigma_sq, z_x, x_tilde, l_x_tilde, x_p, d_x, l_x, d_x_p, z_p = inference(next_batch)

                    # Calculate the loss for this tower
                    SSE_loss, KL_loss, D_loss, G_loss, LL_loss = loss(next_batch, x_tilde, z_x_log_sigma_sq, z_x_mean, d_x, d_x_p, l_x, l_x_tilde, dim1, dim2, dim3)

                    # specify loss to parameters
                    params = tf.trainable_variables()
                    E_params = [i for i in params if 'enc' in i.name]
                    G_params = [i for i in params if 'gen' in i.name]
                    D_params = [i for i in params if 'dis' in i.name]

                    # Calculate the losses specific to encoder, generator, decoder
                    L_e = tf.clip_by_value(KL_loss*KL_param + LL_loss, -100, 100)
                    L_g = tf.clip_by_value(LL_loss*LL_param+ tf.maximum(constraint_boundary, G_loss) * G_param, \
                                           -100, 100)
                    L_d = tf.clip_by_value(D_loss, -100, 100)


                    # Reuse variables for the next tower.

                    # Calculate the gradients for the batch of data on this CIFAR tower.
                    grads_e = opt_E.compute_gradients(L_e, var_list = E_params)
                    grads_g = opt_G.compute_gradients(L_g, var_list = G_params)
                    grads_d = opt_D.compute_gradients(L_d, var_list = D_params)

                    # Keep track of the gradients across all towers.
                    tower_grads_e.append(grads_e)
                    tower_grads_g.append(grads_g)
                    tower_grads_d.append(grads_d)
    # Average the gradients
    grads_e = average_gradients(tower_grads_e)
    grads_g = average_gradients(tower_grads_g)
    grads_d = average_gradients(tower_grads_d)

    # apply the gradients with our optimizers
    #with tf.variable_scope(tf.get_variable_scope(),reuse=False):
    train_E = opt_E.apply_gradients(grads_e, global_step=global_step)
    train_G = opt_G.apply_gradients(grads_g, global_step=global_step)
    train_D = opt_D.apply_gradients(grads_d, global_step=global_step)
    # Start the Session
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() # initialize network saver
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    saver.restore(sess, tf.train.latest_checkpoint(dir_name + '/'))

    
    print("Start iterations")
    epoch = 10
    #tf.train.Saver.restore(saver, sess, 'models/faces_multiGPU_64_0000.tfmod')

    total_batch = int(np.floor(num_examples / (batch_size*num_cpus))) # how many batches are in an epoch

    # We balance of generator and discriminators learning rate by using a sigmoid function,
    #  encouraging the generator and discriminator be about equal
    d_real = 0
    d_fake = 0

    while epoch < num_epochs:
        for i in tqdm.tqdm(range(total_batch)):

            iter_ = data_iterator()
            # balence gen and descrim
            e_current_lr = e_learning_rate*sigmoid(np.mean(d_real),-.5,15)
            g_current_lr = g_learning_rate*sigmoid(np.mean(d_real),-.5,15)
            d_current_lr = d_learning_rate*sigmoid(np.mean(d_fake),-.5,15)
            next_batches, _ = iter_.next()
            _, _, _, D_err, G_err, KL_err, SSE_err, LL_err, d_fake,d_real =\
            sess.run([
                    train_E, train_G, train_D,
                    D_loss, G_loss, KL_loss, SSE_loss, LL_loss,
                    d_x_p, d_x,

                ],
                                            {
                    lr_E: e_current_lr,
                    lr_G: g_current_lr,
                    lr_D: d_current_lr,
                    all_input: next_batches,
                    KL_param: 1,
                    G_param: 1,
                    LL_param: 1
                }
            )
            #KL_err= SSE_err= LL_err = 1
            # Save our lists
            dxp_list.append(d_fake)
            dx_list.append(d_real)
            G_loss_list.append(G_err)
            D_loss_list.append(D_err)
            KL_loss_list.append(KL_err)
            SSE_loss_list.append(SSE_err)
            LL_loss_list.append(LL_err)
            if i%300 == 0:
                # print display network output
                print('Epoch: '+str(epoch))

        # save network
        saver.save(sess,''.join([dir_name +'/faces_multiGPU_64_',\
                              str(epoch).zfill(4),'.tfmod']))
        epoch +=1

