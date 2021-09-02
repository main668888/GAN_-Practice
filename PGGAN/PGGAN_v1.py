# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 19:07:48 2021

@author: user
"""
from keras.models import Model,Sequential
from keras.layers import Input
from keras.layers import Dense,Conv2D,AveragePooling2D,UpSampling2D,Reshape
from keras.layers import Flatten
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.layers import Layer
from keras.layers import Add
from numpy.random import randint
from numpy.random import randn
from keras.initializers import RandomNormal
from keras.constraints import max_norm
from keras import backend
from keras.utils.vis_utils import plot_model
from numpy import load
import numpy as np
from skimage.transform import resize
from matplotlib import pyplot
from numpy import asarray
from math import sqrt
# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

class MinibatchStdev(Layer):
	# initialize the layer
	def __init__(self, **kwargs):
		super(MinibatchStdev, self).__init__(**kwargs)
 
	# perform the operation
	def call(self, inputs):
		# calculate the mean value for each pixel across channels
		mean = backend.mean(inputs, axis=0, keepdims=True)
		# calculate the squared differences between pixel values and mean
		squ_diffs = backend.square(inputs - mean)
		# calculate the average of the squared differences (variance)
		mean_sq_diff = backend.mean(squ_diffs, axis=0, keepdims=True)
		# add a small value to avoid a blow-up when we calculate stdev
		mean_sq_diff += 1e-8
		# square root of the variance (stdev)
		stdev = backend.sqrt(mean_sq_diff)
		# calculate the mean standard deviation across each pixel coord
		mean_pix = backend.mean(stdev, keepdims=True)
		# scale this up to be the size of one input feature map for each sample
		shape = backend.shape(inputs)
		output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
		# concatenate with the output
		combined = backend.concatenate([inputs, output], axis=-1)
		return combined
 
	# define the output shape of the layer
	def compute_output_shape(self, input_shape):
		# create a copy of the input shape as a list
		input_shape = list(input_shape)
		# add one to the channel dimension (assume channels-last)
		input_shape[-1] += 1
		# convert list to a tuple
		return tuple(input_shape)

# weighted sum output
class WeightedSum(Add):
	# init with default value
	def __init__(self, alpha=0.0, **kwargs):
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = backend.variable(alpha, name='ws_alpha')

	# output a weighted sum of inputs
	def _merge_function(self, inputs):
		# only supports a weighted sum of two inputs
		assert (len(inputs) == 2)
		# ((1-a) * input1) + (a * input2)
		output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
		return output

class PixelNormalization(Layer):
	# initialize the layer
	def __init__(self, **kwargs):
		super(PixelNormalization, self).__init__(**kwargs)
 
	# perform the operation
	def call(self, inputs):
		# calculate square pixel values
		values = inputs**2.0
		# calculate the mean pixel values
		mean_values = backend.mean(values, axis=-1, keepdims=True)
		# ensure the mean is not zero
		mean_values += 1.0e-8
		# calculate the sqrt of the mean squared value (L2 norm)
		l2 = backend.sqrt(mean_values)
		# normalize values by the l2 norm
		normalized = inputs / l2
		return normalized
	# define the output shape of the layer
	def compute_output_shape(self, input_shape):
		return input_shape


def add_discriminator_block(old_model,n_input_layers=3):
    # Weight initialization
    init = RandomNormal(stddev=0.02)
    # Weight constraint
    const = max_norm(1.0)
    # get shape of existing model
    in_shape = list(old_model.input.shape)
    input_shape = (in_shape[-3].value*2,in_shape[-2].value*2,in_shape[-1].value)
    in_image = Input(shape = input_shape)
    # define new input processing layer
    d = Conv2D(128,(1,1),padding='same',kernel_initializer=init,kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)  
    # define new block
    d = Conv2D(128,(3,3),padding='same',kernel_initializer=init,kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128,(3,3),padding='same',kernel_initializer=init,kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = AveragePooling2D()(d)
    block_new = d
    # skip the input,1X1 and activation for the old model
    for i in range(n_input_layers,len(old_model.layers)):
        d = old_model.layers[i](d)
    # define straight-through model
    model1 = Model(in_image,d)
    # compile model
    model1.compile(optimizer =  Adam(lr=0.001,beta_1=0,beta_2=0.99,epsilon=10e-8),
                   loss = wasserstein_loss)
    
    # downsample the new larger image
    downsample = AveragePooling2D()(in_image)
    # connect old input processing to downsampled new input
    block_old = old_model.layers[1](downsample)
    block_old = old_model.layers[2](block_old)
    # fade in output of old model input layer with new input
    d = WeightedSum()([block_old,block_new])
    # skip the input,1X1 and activation for the old model
    for i in range(n_input_layers,len(old_model.layers)):
        d = old_model.layers[i](d)
    model2 = Model(in_image,d)
    model2.compile(optimizer=Adam(lr=0.001,beta_1=0,beta_2=0.99,epsilon=10e-8),
                   loss = wasserstein_loss)
    return [model1,model2]
    

def define_discriminator(n_blocks=2,input_shape = (4,4,3)):
    # Weight initialization
    init = RandomNormal(stddev=0.02)
    # Weight constraint
    const = max_norm(1.0)
    model_list = list()
    # base model input
    in_image = Input(shape = input_shape)
    # conv 1X1
    d = Conv2D(128,(1,1),padding='same',kernel_initializer=init,kernel_constraint=const)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 3X3 (output block)
    d = MinibatchStdev()(d)
    d = Conv2D(128,(3,3),padding='same',kernel_initializer=init,kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # conv 4X4
    d = Conv2D(128,(4,4),padding='same',kernel_initializer=init,kernel_constraint=const)(d)
    d = LeakyReLU(alpha=0.2)(d)
    # dense output layer
    d = Flatten()(d)
    out_class = Dense(1)(d)
    # define model
    model = Model(in_image,out_class)
    # compile model
    model.compile(optimizer = Adam(lr=0.001,beta_1=0,beta_2=0.99,epsilon=10e-8),
                  loss = wasserstein_loss)
    # store model
    model_list.append([model,model])
    #create submodels
    for i in range(1,n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i-1][0]
        models = add_discriminator_block(old_model)
        model_list.append(models)
    print("over")
    return model_list
def add_generator_block(old_model):
    print("")
    # weight initializtion
    init = RandomNormal(stddev=0.02)
    # Weight constraint
    const = max_norm(1.0)
    # get the end of the last block
    block_end = old_model.layers[-2].output
    print('block_end:',block_end)
    # upsample, and define new block
    upsampling = UpSampling2D()(block_end)
    g = Conv2D(128,(3,3),padding='same',kernel_initializer=init,kernel_constraint=const)(upsampling)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(128,(3,3),padding='same',kernel_initializer=init,kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU()(g)
    # add new output layer
    out_image = Conv2D(3,(1,1),padding='same',kernel_initializer=init,kernel_constraint=const)(g)
    print("old_model.input:",old_model.input)
    # define model
    model1 = Model(old_model.input,out_image)
    model1.summary()
    # get the output layer from old model
    out_old = old_model.layers[-1]
    # connect the upsampling to the old output layer
    out_image2 = out_old(upsampling)
	# define new output image as the weighted sum of the old and new models
    merged = WeightedSum()([out_image2, out_image])
    #define model
    model2 = Model(old_model.input,merged)
    plot_model(model2,to_file='model2.png',show_shapes=True)
    return [model1,model2]
def define_generator(latent_dim,n_blocks,in_dim=4):
    # weight initializtion
    init = RandomNormal(stddev=0.02)
    # Weight constraint
    const = max_norm(1.0)
    model_list = list()
    # base model input
    in_latent = Input(shape = (latent_dim,))
    # linear scale up to to activation maps
    g  = Dense(128 * in_dim * in_dim, kernel_initializer=init, kernel_constraint=const)(in_latent)
    g = Reshape((in_dim, in_dim, 128))(g)
	# conv 4x4, input block
    g = Conv2D(128, (4,4), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
	# conv 3x3
    g = Conv2D(128, (3,3), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
    g = PixelNormalization()(g)
    g = LeakyReLU(alpha=0.2)(g)
	# conv 1x1, output block
    out_image = Conv2D(3, (1,1), padding='same', kernel_initializer=init, kernel_constraint=const)(g)
	# define model
    model = Model(in_latent,out_image)
    # store model
    model_list.append([model,model])
    # create submodels
    for i in range(1,n_blocks):
        # get prior model without the fade-on
        old_model = model_list[i-1][0]
        # create new model for next resoluation
        models = add_generator_block(old_model)
        model_list.append(models)
    return model_list
def define_composite(discriminators,generators):
    model_list = list()
    # create composit models
    for i in range(len(discriminators)):
        g_models,d_models = generators[i],discriminators[i]
        # straight-through model
        d_models[0].trainable = False
        model1 = Sequential()
        model1.add(g_models[0])
        model1.add(d_models[0])
        model1.compile(optimizer = Adam(lr=0.001,beta_1=0,beta_2=0.99,epsilon=10e-8),
                  loss = wasserstein_loss)
        # fade-in model
        d_models[1].trainable = False
        model2 = Sequential()
        model2.add(g_models[1])
        model2.add(d_models[1])
        model2.compile(optimizer = Adam(lr=0.001,beta_1=0,beta_2=0.99,epsilon=10e-8),
                  loss = wasserstein_loss)
        model_list.append([model1,model2])
    return model_list
def load_real_samples(filename):
    # load dataset
    data = load(filename)
    # extract numpy array
    X = data['arr_0']
    # convert from int to floats
    X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X-127.5)/127.5
    return X
def generators_real_samples(dataset,n_samples):
    # choose random instances
    ix = randint(0,dataset.shape[0],n_samples)
    # selects images
    X = dataset[ix]
    # generate class labels
    y = np.ones((n_samples,1))
    return X,y
def generators_latent_points(latent_dim,n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for network
    x_input = x_input.reshape(n_samples,latent_dim)
    return x_input
def generate_fake_samples(generator,latent_dim,n_samples):
    # generate points in latent space
    x_input = generators_latent_points(latent_dim,n_samples)
    # predict outputs
    X = generator.predict(x_input)
    # create class labels
    y = -np.ones((n_samples,1))
    return X,y
def update_fadein(models,step,n_steps):
    # calculate current alpha (linear from 0 to 1)
    alpha = step / float(n_steps - 1)
    # update the alpha for each model
    for model in models:
        for layer in model.layers:
            if isinstance(layer,WeightedSum):
                backend.set_value(layer.alpha,alpha)
def train_epochs(g_model, d_model, gan_model, dataset, n_epochs, n_batch, fadein=False):
    print("")
    bat_per_epo = int(dataset.shape[0]/n_batch)
    n_steps = bat_per_epo * n_epochs
    half_batch = int(n_batch / 2)
    models = [g_model, d_model, gan_model]
    # manually enumerate epochs
    for i in range(n_steps):
        # update alpha for all WeightedSun layers when fading in new blocks
        if fadein:
            update_fadein(models, i, n_steps)
        # prepare real and fake samples
        X_real,y_real = generators_real_samples(dataset, half_batch)
        X_fake,y_fake = generate_fake_samples(g_model,latent_dim, half_batch)
        # update discriminator model
        d_loss1 = d_model.train_on_batch(X_real,y_real)
        d_loss2 = d_model.train_on_batch(X_fake,y_fake)
        # update the generator via the discrimunator's error
        z_input = generators_latent_points(latent_dim, n_batch)
        y_real2 = np.ones((n_batch,1))
        g_loss = gan_model.train_on_batch(z_input,y_real2)
        # summarize loss on this batch
        print('>%d, d1=%.3f, d2=%.3f g=%.3f' % (i+1, d_loss1, d_loss2, g_loss))
def scale_dataset(images,new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image,new_shape,0)
        # store
        images_list.append(new_image)
    return asarray(images_list)

def summarize_performance(status, g_model, latent_dim, n_samples=25):
	# devise name
	gen_shape = g_model.output_shape
	name = '%03dx%03d-%s' % (gen_shape[1], gen_shape[2], status)
	# generate images
	X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# normalize pixel values to the range [0,1]
	X = (X - X.min()) / (X.max() - X.min())
	# plot real images
	square = int(sqrt(n_samples))
	for i in range(n_samples):
		pyplot.subplot(square, square, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X[i])
	# save plot to file
	filename1 = 'images/plot_%s.png' % (name)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'models/model_%s.h5' % (name)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))
    
def train(g_models, d_models, gan_models, dataset, latent_dim, e_norm, e_fadein, n_batch):
    # fit the baseline model
    g_normal,d_normal,gan_normal = g_models[0][0],d_models[0][0],gan_models[0][0]
    # scale dataset to appropriate size
    gen_shape = g_normal.output_shape
    scaled_data = scale_dataset(dataset, gen_shape[1:])
    print('Scaled Data', scaled_data.shape)
    # train normal or straight-through models
    train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[0], n_batch[0])
    summarize_performance('tuned',g_normal,latent_dim)
    # process each level of growth
    for i in range(1,len(g_models)):
        # retrieve models for this level of growth
        [g_normal,g_fadein] = g_models[i]
        [d_normal,d_fadein] = d_models[i]
        print('gan_models:',len(gan_models))
        [gan_normal,gan_fadein] = gan_models[i]
        # scale dataset to appropriate size
        gen_shape = g_normal.output_shape
        scaled_data = scale_dataset(dataset, gen_shape[1:])
        print("Scaled Data",scaled_data.shape)
        # train fade-in models for next level of growth
        train_epochs(g_fadein, d_fadein, gan_fadein, scaled_data, e_fadein[i], n_batch[i],True)
        summarize_performance('faded',g_fadein,latent_dim)
        train_epochs(g_normal, d_normal, gan_normal, scaled_data, e_norm[i], n_batch[i])
        summarize_performance('trued',g_normal,latent_dim)

if __name__ == '__main__':
    # number of growth phases eq. 6 = [4,8,16,32,64,128]
    n_blocks = 6
    # size of the latent space
    latent_dim = 100
    # define models
    d_models = define_discriminator(n_blocks)
    # define models
    g_models = define_generator(latent_dim,n_blocks)
    # define models
    gan_models = define_composite(d_models, g_models)
    # load image data
    print(np.shape(gan_models))
    runn =True
    if runn ==True:
        dataset = load_real_samples('img_align_gan.npz')
        print('Loaded', dataset.shape)
        # train model
        n_batch = [16,16,16,8,4,4]
        # 10 epochs == 500k images per training phase
        n_epochs = [5,8,8,10,10,10]
        train(g_models, d_models, gan_models, dataset, latent_dim, n_epochs, n_epochs, n_batch)
    else:
        for i in range(2):
            for j in range(6):
                x,z = str(i),str(j)
                plot_model(gan_models[j][i],to_file='model_images_gan/'+z+x+'c.png',show_shapes=True)
                plot_model(d_models[j][i],to_file='model_images_d/'+z+x+'c.png',show_shapes=True)
                plot_model(g_models[j][i],to_file='model_images_g/'+z+x+'c.png',show_shapes=True)
                
# model.summary()

# model = define_discriminator()
# for i in range(2):
#     for x in range(2):
#         print('i:',i,'x:',x)
#         print(model[x][i].summary())
#         j,z = str(i),str(x)
#         plot_model(model[x][i],to_file=j+z+'c.png',show_shapes=True)
        
        
        