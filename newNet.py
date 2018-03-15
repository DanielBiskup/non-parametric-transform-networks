import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
#from rot_conv_try import get_rotated_kernels


import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F

from three_layered_models import threeLayeredCNN, threeLayeredNPTN
from torch.nn.functional import affine_grid, grid_sample
from filter_visualization import plot_kernels, example_plots_cnn, example_plots_nptn
from filter_visualization import imshow_cifar, imshow_mnist
from filter_visualization import show_cifar, show_mnist
import torch.nn as nn
from torch.nn.parameter import Parameter
from network import make_permutation
import torch.optim as optim


def make_permutation(M,N):
    nums = [i for i in range(M*N)]
    NUMS = np.array(nums)
    NUMS = np.reshape(NUMS, (M,N))
    NUMS = NUMS.T
    return NUMS.flatten()

def make_rotation_matrix(rot_deg):
    rot_rad = np.radians(rot_deg)
    c = np.cos(rot_rad)
    s = np.sin(rot_rad)
    mat = np.array([[c,-s,0],[s,c,0]])
    return mat

def make_rotation_batch(rot_deg, batch_size):
    return np.array([make_rotation_matrix(rot_deg) for i in range(batch_size)])

def make_rotations(rot_deg_min, rot_deg_max, num_rots):
    return np.array([make_rotation_matrix(rot_deg) for rot_deg in np.linspace(rot_deg_min, rot_deg_max, num_rots)])

class NewNPTN(nn.Module):
    def __init__(self, M, N, G, alpha, filtersize = 5, padding=0):
        super(NewNPTN, self).__init__()
        
        # M: number of input channeles
        # N: number of output channels
        # G: number of translations to use. If odd, the original filter will also be used. If even, not.
        # alpha: There will be G filters rotated from -alpha to alpha.
        self.M=M
        self.N=N 
        self.G=G
        self.a=alpha
        self.k = filtersize
        
        #self.conv1 = nn.Conv2d(self.M, self.M*self.N*self.G, filtersize, groups=self.M, padding=padding) # in, out, kernel size, groups as in paper
        
        # Lets follow the vocabluary of the paper and call:
        # w: the unrotated filters / "templates" (the ones we want to learn)
        # gw: the rotated filters / "transformed templates" (the G rotated versions of those filters)
        # There will be a total of:
        #   M*N templates
        #   M*N*G transformed templates
 
        # Create the "templates" as torch.nn.Parameter as described here:
        # http://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html
        # http://pytorch.org/docs/0.3.1/nn.html#parameters

        # Rondomly initialize weights: # TODO: Maybe there is a better way to initialize
        # maybe: #w = Parameter(torch.randn(M*N, k, k))  # size(w) = ( M*N x k x k )

       
        self.w = Parameter(torch.randn(M*N, 1, k, k))  # size(w) = ( M*N x 1 x k x k )
        
        # From every "template" derive G "transformed templates" by applying rotation:
        # Variable(torch.randn(8,4,3,3))
        # Step 1 – Create the G transformation matrices we whish to rotate each "template" by:
        #g = get_rotated_kernels(w, self.G, -alpha, alpha) #Variable(torch.randn(G,2,3)) # size(g) = ( G x 2 x 3 ) # TODO

        #g = Variable(torch.randn(G,2,3)) # size(g) = ( G x 2 x 3 ) # TODO
        g1 = make_rotations(-alpha, alpha, G) # size(g) = ( G x 2 x 3 )
        g2 = [torch.Tensor(f) for f in g1]
        g3 = torch.stack(g2)
        g = Variable( torch.Tensor(  g3  ))
        
        # Step2 – Apply the transformations:
        # Step 2.1 – Create the flow fields, describing the transformations.
        s = torch.Size((G, M*N, k, k)) # Desired output size
        flow_field  = torch.nn.functional.affine_grid(g, s) # size(flow_field) = (G, k, k, 2), one translation vector (or maybe coordinate; we don't know nor care) per each of the G rotation matrices.
        self.register_buffer("flow_field", flow_field)
        
        # Those two layers are the same as in the vanilla NPTN by ??? et.al. 20??
        self.maxpool3d = nn.MaxPool3d((self.G, 1, 1)) 
        self.meanpool3d = nn.AvgPool3d((self.M, 1, 1)) # Is that the right pooling? - AvgPool3d?
        
        self.permutation = make_permutation(self.M, self.N)

    def forward(self, x):
        # Start from w ervery time and create the others as rotation of it
        
        #print('\nShape of x ', x.size())
       
        # Step 2.2 – Apply the flow_fields. Each flow_field will be applied to each channle of the input / each "template".
        # Each flow_field is of (G x M*N, k, k)
        #a grid. For each rotation, there will be one flow field.
        # Repeat w along the first dimension:
        
        # Permute:  (M*N, 1, k, k) to (1, M*N, k, k)
        w_rep = self.w.permute(1,0,2,3)
        # ( 1 x M*N, k, k) to ( G x M*N, k, k)
        w_rep2 = w_rep.expand(G, -1, -1, -1) # Repeat along the singular first dimension G times.
        #                                    # size(w_rep2) = ( G x M*N, k, k)
        # 
        w_rot = torch.nn.functional.grid_sample(w_rep2, self.flow_field) # size(w_rep2) = ( G x M*N x k x k )
        
        # Go from ( G x M*N x k x k ) to ( M*N x G x k x k ),
        #     i.e.( angle, template, x, y) to ( template, angle, x, y)
        w_perm = w_rot.permute(1,0,2,3)
        
        # But actually we need this unrolled:
        MNG = self.M * self.N * self.G
        
        w_unrolled = w_perm.resize(MNG, 1, self.k, self.k) # size(w_unrolled) = ( M*N*G x (M*N*G)/(M*N*G) x k x k ) #weight – filters of shape (out_channels×in_channelsgroups×kH×kW)
        
        # Convolution:
        # Use the functional (torch.nn.functional.conv2d) instead of Module
        # (torch.nn.conv2d), becuse we don't want trainable parameters exept the
        # template vector defined above:
        x = F.conv2d(x, w_unrolled, groups=M)  # TODO: Add padding?        
                
        #print('Shape after convolution', x.size())
        x = self.maxpool3d(x)
        #print("Shape after MaxPool3d: ", x.size()) # dimension should be M*N
        
        #print('permutation ', permutation)
        x = x[:, self.permutation] # reorder channels
        #print("Shape after Channel reordering: ", x.size())
        x = self.meanpool3d(x)
        #print('Shape after Mean Pooling: ', x.size())
        return x


class TestNet(nn.Module):
    def __init__(self, M, N, G, rot_min = -90, rot_max = +90, filtersize = 3, padding=0):
        super(TestNet, self).__init__()

        self.M=M
        self.N=N
        self.G=G
        self.rot_min = rot_min
        self.rot_max = rot_max
        self.k = filtersize

        if M==3: # CIFAR
            self.input_size=(3,32,32)
        else:
            self.input_size=(1,28,28)

        #padding = int(filtersize/2) # do or don't ?
        
        # first layer 
        self.newNet = NewNPTN(M, N, G, alpha = rot_max, filtersize = k, padding=0)
        n = self.num_flat_features(self.input_size)
        
        self.fc1 = nn.Linear(n, 10)
        
    def num_flat_features(self, input_size):
        t = Variable(torch.ones(1, *input_size))
        #print('t.size() = ' + str(t.size()))
        f = self.features(t)
        #print('Shape after convolution layers = ' + str(f.size()))
        n = int(np.prod(f.size()[1:]))
        return n  
    
    def features(self, x):
        # first layer
        x = self.newNet(x)
        return x
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.log_softmax(self.fc1(x), dim=1)
        #print('shape prediction',x.shape)
        return x

#####  small run to see if weights are updated  #####

use_cuda = True
plot = True

# load MNIST data and rotate it
max_rotation = 0
batch_size = 32
# Training dataset
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform=transforms.Compose([
                       #torchvision.transforms.RandomCrop((32,32), padding=2),    
                       torchvision.transforms.RandomRotation(max_rotation),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=batch_size, shuffle=True, num_workers=4)
# Test dataset
testloader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False, transform=transforms.Compose([
        torchvision.transforms.RandomRotation(max_rotation),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=batch_size, shuffle=False, num_workers=4)


# DEV_VALUES__
M = 1 # M: number of input channeles
N = 2 # N: number of output channels
G = 5 # G: number of translations to use.
alpha = 5
k = 3 # filtersize
# inputs = Variable(torch.randn(G,2,3)) # size(g) = ( G x 2 x 3 ) # TODO
#__DEV_VALUES

#net = NewNPTN(M, N, G, alpha = alpha, filtersize = k, padding=0)
net = TestNet( M, N, G, alpha, filtersize = k, padding=0)

if use_cuda:
    net.cuda()
#init_w = net.rotPTN.rotConv.w.detach() # does not work since copy is not possible
#print(net.rotPTN.rotConv.w)   

if plot:
    plot_kernels(net.newNet.w, title='initialization, layer 1')
    #plot_kernels(net.rotPTN2.rotConv.w, title='initialization, layer 2')
criterion = nn.NLLLoss() 
optimizer = optim.SGD(net.parameters(), lr=0.05)



def training_epoch(epoch):
    running_loss = 0.0
    correct = 0
    
    for i, data in enumerate(trainloader, 0): 
         
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if use_cuda:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.data).sum()
        
        if i % 250 == 249:
            stat_epoch = epoch + 1
            stat_batch = i + 1
            stat_loss = running_loss / 250
            print('[%d, %5d] loss: %.3f' %
                  (stat_epoch, stat_batch, stat_loss))
          
            running_loss = 0.0
            
    accuracy = (100 * correct / trainloader.dataset.train_data.shape[0])

    print('----------------------------------------------')
    print('Epoch ', epoch) 
    print('Accuracy of the network on the train images: %d %%' % accuracy)
    if plot:
        plot_kernels(net.newNet.w, title='layer 1')
        #plot_kernels(net.rotPTN2.rotConv.w, title='layer 2')

num_epochs = 3 # paper: 300
for epoch in range(num_epochs):  # loop over the dataset multiple times

    training_epoch(epoch)
    
end_w = net.newNet.w


