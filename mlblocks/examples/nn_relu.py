from mlblocks.gradient import TensorValue, Neuron
from mlblocks.layers import LinearLayer
from mlblocks.arch import AbstractBaseNet
from mlblocks.dataset import AbstractDataHolder
import struct
import random
import array
import os
    

class DataHolder(AbstractDataHolder):
    def __init__(self):
        self.image_dimension = (28, 28)
        self.train_img = self.extract_images("./Dataset/train-images.idx3-ubyte")
        self.train_label = self.extract_labels("./Dataset/train-labels.idx1-ubyte")
        self.test_img = self.extract_images("./Dataset/t10k-images.idx3-ubyte")
        self.test_label = self.extract_labels("./Dataset/t10k-labels.idx1-ubyte")
    
    def extract_labels(self, path):
        res = []
        with open(path, 'rb') as fp:
            # extract header
            _, size = struct.unpack("!II", fp.read(4*2))
            res = array.array("B", fp.read(size))
        return res
    
    def extract_images(self, path):
        res = []
        with open(path, 'rb') as fp:
            magic_code, size, rows, cols = struct.unpack("!IIII", fp.read(4*4))
            dim = [size, rows, cols]
            buffer_count = cols*rows
            image_data = array.array("B", fp.read())
            for index in range(size):
                raw_arr = image_data[index*buffer_count:(index*buffer_count)+buffer_count]
                res.append(raw_arr)
        return res
    
    def get_dataset(self, index):
        return self.train_img[index].tolist(), self.train_label[index]

    def get_test_dataset(self, index):
        return self.test_img[index].tolist(), self.test_label[index]
    
    def get_img(self, index):
        img_arr, label = self.get_dataset(index)
        image = []
        for w in range(28):
            image.append(img_arr[w*28:(w+1)*28])
        return image, label
    
    def prev_img(self, index):
        image, label = self.get_dataset(index)
        plt.imshow(image, cmap='gray')
        plt.axis('off') 
        plt.title(label)
        plt.show()
        return image, label



class BNet(AbstractBaseNet):
    def __init__(self):
        # 728 -> 10 -> 10
        
        self.layer_one = LinearLayer(
            [Neuron(random.uniform(-0.1, 0.1) for _ in range(728)) for _ in range(10)], 
            label="hidden")
        
        self.layer_two = LinearLayer(
            [Neuron(random.uniform(-0.1, 0.1) for _ in range(10)) for _ in range(10)], 
            label="output")
        
        self.pre_feed_hook = lambda arr: [v/255 for v in arr]
        self.post_feed_hook = self.softmax
        # register layers
        self.layers.append(self.layer_one)
        self.layers.append(self.layer_two)

    def predict(self, input_array):
        arr = self.feed(input_array)
        c = dict(zip(range(9), [i.num for i in arr]))
        return c, max(c, key=c.get)
    
    def get_loss(self, input_img, label):
        target = [0]*10
        # our expectations
        target[label]=1
        # prediction
        arr = self.feed(input_img)
        # calculating mean squared error
        loss = self.mean_square_error(arr, target)
        c = dict(zip(range(9), [i.num for i in arr]))
        return loss, max(c, key=c.get)
# dataset
dt_obj = DataHolder()
# nn
nn = BNet()
learning_rate = 0.01
dataset_index = list(range(59990))
random.shuffle(dataset_index)

# pre-trained weight_file
weight_file = "./nn_relu.wt"
Train = False
if os.path.exists(weight_file):
    nn.load(weight_file, LinearLayer, Neuron, TensorValue)

# loop
for datasetIndex in dataset_index:
    img_arr, actual_label = dt_obj.get_dataset(datasetIndex)
    if Train:
        (loss_value, predict_value) = nn.get_loss(img_arr, actual_label)
        # calculating backward gradient
        loss_value.backward()

        for w in nn.get_parameters():
            w.num += (w.grad * learning_rate * -1)
        (new_loss, predict_value)= nn.get_loss(img_arr, actual_label)
        print(f"datasetIndex : {datasetIndex}, pre_loss:{loss_value.num}, now_loss:{new_loss.num}, predict:{predict_value}, actual:{actual_label}")
    else:
        _, predict_value= nn.predict(img_arr)
        print(f"datasetIndex : {datasetIndex}, predict:{predict_value}, actual:{actual_label}")
        
