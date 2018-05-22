"""://mxnet.incubator.apache.org/versions/master/api/python/gluon/model_zoo.html
The list of available gluon models are here
https://mxnet.incubator.apache.org/versions/master/api/python/gluon/model_zoo.html

So far, we confirmed that the model alexnet, resnet18_v1, resnet50_v1 can be loaded to nnvm
Other models do not work with NNVM at the moment.

See this issue https://github.com/dmlc/nnvm/issues/203.

"""

import mxnet as mx
import nnvm
import tvm
import numpy as np

from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
from PIL import Image
from matplotlib import pyplot as plt

#model = "resnet50_v1"
model = "mobilenet1.0"
#model = "inceptionv3"
print("Testing model %s" % model)
block = get_model(model, pretrained=True)
img_name = 'data/cat.png'
synset_name = 'data/synset.txt'
with open(synset_name) as f:
    synset = eval(f.read())
image = Image.open(img_name).resize((224, 224))
# plt.imshow(image)
# plt.show()

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis]
    return image

x = transform_image(image)
print('x', x.shape)

######################################################################
# Compile the Graph
# -----------------
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.
# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
sym, params = nnvm.frontend.from_mxnet(block)
# we want a probability so add a softmax operator
sym = nnvm.sym.softmax(sym)

######################################################################
# now compile the graph
import nnvm.compiler
target = 'opencl'
shape_dict = {'data': x.shape}
graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
from tvm.contrib import graph_runtime
ctx = tvm.opencl()
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)

# set inputs
arr = tvm.nd.array(x.astype(dtype))
m.set_input('data', arr)
m.set_input(**params)
# execute
m.run()
import time
s = time.time()
for n in range(1024):
    m.run()
    ctx.sync()
tt = (time.time() - s)/1024
print("Per example: {}, inferences/sec: {}".format(tt, 1.0/tt))
