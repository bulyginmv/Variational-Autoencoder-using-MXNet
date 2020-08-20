from mxnet import autograd, nd, gluon, init, nd, io
from mxnet.gluon import nn

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np

class Decoder(nn.HybridSequential):
    def __init__(self, activation = "relu", hiddens = 400, observables = 784, layers = 1, **kwargs): # Feel free to use different arguments
        super(Decoder, self).__init__(**kwargs)
        with self.name_scope():
            for i in range(layers):
                self.add(nn.Dense(hiddens, activation = activation))
            self.add(nn.Dense(observables, activation='sigmoid'))


class Encoder(nn.HybridSequential):
    def __init__(self, activation = "relu", hiddens= 400, latents = 2, layers = 1, **kwargs): # Feel free to use different arguments
        super(Encoder, self).__init__(**kwargs)
        with self.name_scope():
            for i in range(layers):
                self.add(nn.Dense(hiddens,activation=activation))
            self.add(nn.Dense(latents*2, activation=None))


class VariationalAutoencoder(gluon.HybridBlock):
    def __init__(self, activation="relu", hiddens=400, latents=2, observables=784, batch_size=100, layers=1,
                 **kwargs):  # Feel free to use different arguments
        self.output = None
        self.mu = None
        self.batch_size = batch_size
        self.latents = latents
        super(VariationalAutoencoder, self).__init__(**kwargs)
        with self.name_scope():
            self.encoder = nn.HybridSequential(prefix='encoder')
            for i in range(layers):
                self.encoder.add(nn.Dense(hiddens, activation=activation))
            self.encoder.add(nn.Dense(latents * 2, activation=None))

            self.decoder = nn.HybridSequential(prefix='decoder')
            for i in range(layers):
                self.decoder.add(nn.Dense(hiddens, activation=activation))
            self.decoder.add(nn.Dense(observables, activation='sigmoid'))
        # self.decoder = Encoder(activation, hiddens, observables, layers,**kwargs)
        # self.encoder = Decoder(activation, hiddens, latents, layers, **kwargs)

    def hybrid_forward(self, F, x):
        #         h = self.encoder(x)
        #         #print(h)
        #         z_mean, z_log_var = F.split(h, axis = 1, num_outputs =2)
        #         eps = F.random_normal(loc=0, scale=1, shape=(x.shape[0], self.latents))
        #         z = F.exp(0.5 * z_log_var) * eps + z_mean
        #         mu_lv = F.split(h, axis=1, num_outputs=2)
        #         x_mean = mu_lv[0]
        #         self.x_mean = x_mean
        #         x_log_var = mu_lv[1]
        #         #x_mean, x_log_var = nd.split(self.decoder(z), axis=1, num_outputs=2)
        #         x_mean, x_log_var = nd.split(self.decoder(z), 2, 1)

        h = self.encoder(x)
        # print(h)
        mu_lv = F.split(h, axis=1, num_outputs=2)
        mu = mu_lv[0]
        lv = mu_lv[1]
        self.mu = mu
        eps = F.random_normal(loc=0, scale=1, shape=(self.batch_size, self.latents))
        z = mu + F.exp(0.5 * lv) * eps
        y = self.decoder(z)
        self.output = y

        KL = 0.5 * F.sum(1 + lv - mu * mu - F.exp(lv), axis=1)
        logloss = F.sum(x * F.log(y + 1e-10) + (1 - x) * F.log(1 - y + 1e-10), axis=1)
        loss = -logloss - KL

        # return x,x_mean, x_log_var, z_mean, z_log_var
        return loss


batch_size = 100
mnist= mx.test_utils.get_mnist()
train_data = np.reshape(mnist['train_data'],(-1,28*28))
test_data = np.reshape(mnist['test_data'],(-1,28*28))
n_batches = train_data.shape[0]/batch_size
train_iter = mx.io.NDArrayIter(data={'data': train_data}, label={'label': mnist['train_label']}, batch_size = batch_size)
test_iter = mx.io.NDArrayIter(data={'data': test_data}, label={'label': mnist['test_label']}, batch_size = batch_size)


from tqdm import tqdm, tqdm_notebook #nice progress bar
import time
import math

#Load in MNIST
mx.random.seed(42)
batch_size = 100

#model_prefix = 'vae_gluon_{}d{}l{}h.params'.format(n_latent, n_layers, n_hidden,layers=n_layers)

model = VariationalAutoencoder()
model.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
model.hybridize()
trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': .001})
lossfun = Lossfun()

epochs = 50
print_period = epochs // 10
start = time.time()
training_loss = []
test_loss = []
for epoch in tqdm_notebook(range(epochs), desc='epochs'):
    epoch_loss = 0
    epoch_val_loss = 0

    train_iter.reset()
    test_iter.reset()
    n_batch_train = 0
    for batch in train_iter:
        n_batch_train +=1
        data = batch.data[0]
        with autograd.record():
            #loss = Lossfun(data)
            loss = model(data)
        loss.backward()
        trainer.step(data.shape[0])
        epoch_loss += nd.mean(loss).asscalar()

    n_batch_val = 0
    for batch in test_iter:
        n_batch_val +=1
        data = batch.data[0]
        loss = model(data)
        epoch_val_loss += nd.mean(loss).asscalar()

    epoch_loss /= n_batch_train
    epoch_val_loss /= n_batch_val

    training_loss.append(epoch_loss)
    test_loss.append(epoch_val_loss)

    if epoch % max(print_period,1) == 0:
        tqdm.write('Epoch{}, Training loss {:.2f}, Validation loss {:.2f}'.format(epoch, epoch_loss, epoch_val_loss))

end = time.time()
print('Time elapsed: {:.2f}s'.format(end - start))

model.save_parameters('varauto')
net2 = VariationalAutoencoder()
net2.load_parameters('varauto', ctx=mx.cpu())


test_iter.reset()
test_batch = test_iter.next()
net2(test_batch.data[0].as_in_context(mx.cpu()))
result = net2.output.asnumpy()
#result = np.reshape(result,(-1,28*28)).shape
original = test_batch.data[0].asnumpy()


#Plot original vs reconstructed
n_samples = 10
idx = np.random.choice(batch_size, n_samples)
_, axarr = plt.subplots(2, n_samples, figsize=(16,4))
for i,j in enumerate(idx):
    axarr[0,i].imshow(original[j].reshape((28,28)), cmap='Greys')
    if i==0:
        axarr[0,i].set_title('original')
    #axarr[0,i].axis('off')
    axarr[0,i].get_xaxis().set_ticks([])
    axarr[0,i].get_yaxis().set_ticks([])

    axarr[1,i].imshow(result[j].reshape((28,28)), cmap='Greys')
    if i==0:
        axarr[1,i].set_title('reconstruction')
    #axarr[1,i].axis('off')
    axarr[1,i].get_xaxis().set_ticks([])
    axarr[1,i].get_yaxis().set_ticks([])
plt.show()

#Randomly sample from the VAE
n_samples = 10
zsamples = nd.array(np.random.randn(n_samples*n_samples, 2))

images = net2.decoder(zsamples.as_in_context(mx.cpu())).asnumpy()

canvas = np.empty((28*n_samples, 28*n_samples))
for i, img in enumerate(images):
    x = i // n_samples
    y = i % n_samples
    canvas[(n_samples-y-1)*28:(n_samples-y)*28, x*28:(x+1)*28] = img.reshape(28, 28)
plt.figure(figsize=(5, 5))
plt.imshow(canvas, origin="upper", cmap="Greys")
plt.axis('off')
plt.tight_layout()
plt.savefig('generated_samples_with_{}D_latent_space.png'.format(2))

n_batches = 10
counter = 0
results = []
labels = []
for batch in test_iter:
    net2(batch.data[0].as_in_context(mx.cpu()))
    results.append(net2.mu.asnumpy())
    labels.append(batch.label[0].asnumpy())
    counter +=1
    if counter >= n_batches:
        break

result= np.vstack(results)
labels = np.hstack(labels)

if result.shape[1]==2:
    from scipy.special import ndtri
    from scipy.stats import norm

    fig, axarr = plt.subplots(1,2, figsize=(10,4))
    im=axarr[0].scatter(result[:, 0], result[:, 1], c=labels, alpha=0.6, cmap='Paired')
    axarr[0].set_title(r'scatter plot of $\mu$')
    axarr[0].axis('equal')
    fig.colorbar(im, ax=axarr[0])

    im=axarr[1].scatter(norm.cdf(result[:, 0]), norm.cdf(result[:, 1]), c=labels, alpha=0.6, cmap='Paired')
    axarr[1].set_title(r'scatter plot of $\mu$ on norm.cdf() transformed coordinates')
    axarr[1].axis('equal')
    fig.colorbar(im, ax=axarr[1])
    plt.tight_layout()
    plt.savefig('2d_latent_space_for_test_samples.png')

#from gluoncv.data import transforms as gcv_transforms
from mxnet.gluon.data.vision import datasets, transforms

def transform(data, label):
    data = data.astype('float32')/255
    return data, label

train_fashion = mx.gluon.data.vision.datasets.FashionMNIST(train=True, transform=transform)
valid_fashion = mx.gluon.data.vision.datasets.FashionMNIST(train=False, transform=transform)
net3 = VariationalAutoencoder()
net3.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
#net3.load_parameters('varauto', ctx=mx.cpu())
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


from matplotlib.pylab import imshow

sample_idx = 234
sample = train_fashion[sample_idx]
data = sample[0]
label = sample[1]
label_desc = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

imshow(data[:,:,0].asnumpy(), cmap='gray')
print("Data type: {}".format(data.dtype))
print("Label: {}".format(label))
print("Label description: {}".format(label_desc[label]))

batch_size = 100
train_data_loader = mx.gluon.data.DataLoader(train_fashion, batch_size, shuffle=True)
valid_data_loader = mx.gluon.data.DataLoader(valid_fashion, batch_size)


from tqdm import tqdm, tqdm_notebook #nice progress bar
import time
import math

#Load in MNIST
mx.random.seed(42)
batch_size = 100

trainer = gluon.Trainer(net3.collect_params(), 'adam', {'learning_rate': .001})

epochs = 50
print_period = epochs // 10
start = time.time()
training_loss = []
test_loss = []
for epoch in tqdm_notebook(range(epochs), desc='epochs'):
    epoch_loss = 0
    epoch_val_loss = 0

    #train_iter.reset()
    #test_iter.reset()
    n_batch_train = 0
    for (data, label) in train_data_loader:
        n_batch_train +=1
        data = data.as_in_context(mx.cpu()).reshape((-1, 784)) # 28*28=784
        with autograd.record():
            #loss = Lossfun(data)
            loss = net3(data)
        loss.backward()
        trainer.step(data.shape[0])
        epoch_loss += nd.mean(loss).asscalar()

    n_batch_val = 0
    for (data, label) in valid_data_loader:
        n_batch_val +=1
        data = data.as_in_context(mx.cpu()).reshape((-1, 784)) # 28*28=784
        loss = net3(data)
        epoch_val_loss += nd.mean(loss).asscalar()

    epoch_loss /= n_batch_train
    epoch_val_loss /= n_batch_val

    training_loss.append(epoch_loss)
    test_loss.append(epoch_val_loss)

    if epoch % max(print_period,1) == 0:
        tqdm.write('Epoch{}, Training loss {:.2f}, Validation loss {:.2f}'.format(epoch, epoch_loss, epoch_val_loss))

end = time.time()
print('Time elapsed: {:.2f}s'.format(end - start))

result = net3.output.asnumpy()

sample_idx = 234
sample = valid_fashion[sample_idx]
data = sample[0]
label = sample[1]
label_desc = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle boot'}

imshow(data[:,:,0].asnumpy(), cmap='gray')
print("Data type: {}".format(data.dtype))
print("Label: {}".format(label))
print("Label description: {}".format(label_desc[label]))

n_samples = 10
zsamples = nd.array(np.random.randn(n_samples*n_samples, 2))
images = net3.decoder(zsamples.as_in_context(mx.cpu())).asnumpy()

canvas = np.empty((28*n_samples, 28*n_samples))
for i, img in enumerate(images):
    x = i // n_samples
    y = i % n_samples
    canvas[(n_samples-y-1)*28:(n_samples-y)*28, x*28:(x+1)*28] = img.reshape(28, 28)
plt.figure(figsize=(5, 5))
plt.imshow(canvas, origin="upper", cmap="Greys")
plt.axis('off')
plt.tight_layout()
plt.savefig('generated_samples_with_{}D_latent_space.png'.format(2))

n_samples = 10
idx = np.random.choice(batch_size, n_samples)
_, axarr = plt.subplots(2, n_samples, figsize=(16,4))
for i,j in enumerate(idx):
    axarr[0,i].imshow(valid_fashion[j][0][:,:,0].asnumpy(), cmap='Greys')
    if(i ==0):
        axarr[0,i].set_title('Random fashionMNIST image')
#     if i==0:
#         axarr[0,i].set_title('original')
#     #axarr[0,i].axis('off')
#     axarr[0,i].get_xaxis().set_ticks([])
#     axarr[0,i].get_yaxis().set_ticks([])

    axarr[1,i].imshow(result[j].reshape((28,28)), cmap='Greys')
    if i==0:
        axarr[1,i].set_title('reconstruction')
    #axarr[1,i].axis('off')
    axarr[1,i].get_xaxis().set_ticks([])
    axarr[1,i].get_yaxis().set_ticks([])
plt.show()