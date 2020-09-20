######################################################################
# Utils for downloading and extracting zip files
# ---------------------------------------------
import os
import tvm
from tvm import relay
from tvm.relay import transform as _transform
import numpy as np

def extract(path):
    import tarfile
    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError('Could not decompress the file: ' + path)

######################################################################
# Load pretrained TFLite model
# ---------------------------------------------
# we load mobilenet V1 TFLite model provided by Google
from tvm.contrib.download import download_testdata

model_url = "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz"

# we download model tar file and extract, finally get tflite file
model_path = download_testdata(model_url, "mobilenet_v1_1.0_224_quant.tgz", module=['tf', 'official'])
model_dir = os.path.dirname(model_path)
extract(model_path)

# now we have mobilenet_v1_1.0_224_quant.tflite on disk and open it
tflite_model_file = os.path.join(model_dir, "mobilenet_v1_1.0_224_quant.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()

# get TFLite model from buffer
from tflite.Model import Model # edit here
tflite_model = Model.GetRootAsModel(tflite_model_buf, 0)

#######################################################################
# Generic run functions for TVM & TFLite
# --------------------------------------
target = tvm.target.riscv_cpu("spike")
input_tensor = "input"
input_shape = (1, 224, 224, 3)
input_dtype = "uint8"

# Parse TFLite model and convert it to a Relay module
mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: input_shape},
                                         dtype_dict={input_tensor: input_dtype})

""" tensorize flow
    call FTVMQnnLegalize() & FTVMQnnCanonicalize for QNN utilty
    call ConvertLayout() to convert conv2d layout from NHWC to NCHW
    call Legalize() to adding pad to conv2d for match the axis (Input channel : as mutiple of 4 / Output channel : as multiple of 16 )
    call AlterOpLaout() to convert conv2d into conv2d_int8
"""
print('qnn_mod before : ', mod)
seq = tvm.transform.Sequential([
    relay.transform.Legalize('FTVMQnnLegalize'),
    relay.transform.Legalize('FTVMQnnCanonicalize'),
    relay.transform.ConvertLayout({'nn.conv2d': ['NCHW', 'OIHW']}),
    relay.transform.Legalize(),
    relay.transform.AlterOpLayout(),
])
with tvm.transform.PassContext(opt_level=3):
    with tvm.target.create(target):
        mod = seq(mod)
print('qnn_mod after : ', mod)
# --------------------------------------------------------------

# opt pass
from tvm.relay.quantize.quantize import _bind_params
optimize = tvm.transform.Sequential([relay.transform.SimplifyInference(),
                                    relay.transform.FoldConstant(),
                                    relay.transform.FoldScaleAxis(),
                                    relay.transform.CanonicalizeOps(),
                                    relay.transform.FoldConstant()])
mod['main'] = _bind_params(mod['main'], params)
with tvm.transform.PassContext(opt_level=3):
        mod = optimize(mod)
print('opt mod : ', mod)

# --------------------------------------------------------------


with relay.build_config(opt_level=0):
    module = relay.build(mod, target, params=params)

lib, graph, params = module.get_lib(), module.get_json(), module.get_params()

target_dir, model_name = '.', 'mobilenet'

with open(target_dir + '/' + model_name + '.ll', 'w') as _f:
        _f.write(lib.get_source())
with open(target_dir + '/' + model_name + '.graph', 'w') as _f:
        _f.write(graph)
with open(target_dir + '/' + model_name + '.params', 'wb') as _f:
        _f.write(relay.save_param_dict(params))
print("save finish")
