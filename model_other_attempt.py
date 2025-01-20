import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from torchvision.transforms.functional import to_pil_image


class Net(nn.Module):
    def __init__(self, num_classes: int = 500, pretrained_model_name: str = "facebook/dinov2-base-imagenet1k-1-layer", freeze_backbone: bool = True):
        super(Net, self).__init__()
        self.base_model = AutoModel.from_pretrained(pretrained_model_name)
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        self.feature_dim = self.base_model.config.hidden_size

        if freeze_backbone:
            self.base_model.requires_grad_(False)

        # Enable gradient checkpointing for memory optimization
        self.base_model.gradient_checkpointing_enable()

        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        pil_images = [to_pil_image(img) for img in x]
        pixel_values = self.processor(images=pil_images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(next(self.base_model.parameters()).device)

        features = self.base_model(pixel_values).last_hidden_state.mean(dim=1)
        bottleneck_features = self.bottleneck(features)
        return self.classifier(bottleneck_features)


"""2024-12-05 14:29:11.812773: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:485] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2024-12-05 14:29:11.834596: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:8454] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2024-12-05 14:29:11.840915: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1452] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2024-12-05 14:29:11.857106: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-12-05 14:29:13.550394: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
config.json: 100% 548/548 [00:00<00:00, 3.41MB/s]
model.safetensors: 100% 4.55G/4.55G [00:28<00:00, 161MB/s]
preprocessor_config.json: 100% 436/436 [00:00<00:00, 1.81MB/s]
/usr/local/lib/python3.10/dist-packages/torch/utils/checkpoint.py:87: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
Train Epoch: 1 [0/20000 (0%)]	Loss: 6.223877
Train Epoch: 1 [1280/20000 (6%)]	Loss: 6.121216
Train Epoch: 1 [2560/20000 (13%)]	Loss: 5.986908
Train Epoch: 1 [3840/20000 (19%)]	Loss: 5.639793
Train Epoch: 1 [5120/20000 (25%)]	Loss: 5.461684
Train Epoch: 1 [6400/20000 (32%)]	Loss: 4.961294
Train Epoch: 1 [7680/20000 (38%)]	Loss: 4.147863
Train Epoch: 1 [8960/20000 (45%)]	Loss: 3.942903
Train Epoch: 1 [10240/20000 (51%)]	Loss: 3.487423
Train Epoch: 1 [11520/20000 (57%)]	Loss: 3.554981
Train Epoch: 1 [12800/20000 (64%)]	Loss: 3.409540
Train Epoch: 1 [14080/20000 (70%)]	Loss: 2.838558
Train Epoch: 1 [15360/20000 (76%)]	Loss: 2.862428
Train Epoch: 1 [16640/20000 (83%)]	Loss: 2.993649
Train Epoch: 1 [17920/20000 (89%)]	Loss: 2.910627
Train Epoch: 1 [19200/20000 (96%)]	Loss: 2.602292

Train set: Accuracy: 21.52%


Validation set: Loss: 51.1366, Accuracy: 41.60%

Saved new best model with accuracy: 41.60%
Train Epoch: 2 [0/20000 (0%)]	Loss: 2.494694
Train Epoch: 2 [1280/20000 (6%)]	Loss: 2.282715
Train Epoch: 2 [2560/20000 (13%)]	Loss: 2.153769
Train Epoch: 2 [3840/20000 (19%)]	Loss: 1.812848
Train Epoch: 2 [5120/20000 (25%)]	Loss: 1.906351
Train Epoch: 2 [6400/20000 (32%)]	Loss: 1.980124
Train Epoch: 2 [7680/20000 (38%)]	Loss: 1.965461
Train Epoch: 2 [8960/20000 (45%)]	Loss: 1.796383
Train Epoch: 2 [10240/20000 (51%)]	Loss: 1.819348
Train Epoch: 2 [11520/20000 (57%)]	Loss: 1.847720
Train Epoch: 2 [12800/20000 (64%)]	Loss: 1.913352
Train Epoch: 2 [14080/20000 (70%)]	Loss: 1.631615
Train Epoch: 2 [15360/20000 (76%)]	Loss: 1.768169
Train Epoch: 2 [16640/20000 (83%)]	Loss: 1.566053
Train Epoch: 2 [17920/20000 (89%)]	Loss: 1.550216
Train Epoch: 2 [19200/20000 (96%)]	Loss: 1.603374

Train set: Accuracy: 54.60%


Validation set: Loss: 37.0500, Accuracy: 54.88%

Saved new best model with accuracy: 54.88%
Train Epoch: 3 [0/20000 (0%)]	Loss: 1.347765
Train Epoch: 3 [1280/20000 (6%)]	Loss: 1.270038
Train Epoch: 3 [2560/20000 (13%)]	Loss: 1.431246
Train Epoch: 3 [3840/20000 (19%)]	Loss: 1.093842
Train Epoch: 3 [5120/20000 (25%)]	Loss: 1.185402
Train Epoch: 3 [6400/20000 (32%)]	Loss: 0.958313
Train Epoch: 3 [7680/20000 (38%)]	Loss: 1.124194
Train Epoch: 3 [8960/20000 (45%)]	Loss: 1.151677
Train Epoch: 3 [10240/20000 (51%)]	Loss: 1.223039
Train Epoch: 3 [11520/20000 (57%)]	Loss: 1.308231
Train Epoch: 3 [12800/20000 (64%)]	Loss: 0.878417
Train Epoch: 3 [14080/20000 (70%)]	Loss: 1.343057
Train Epoch: 3 [15360/20000 (76%)]	Loss: 1.142106
Train Epoch: 3 [16640/20000 (83%)]	Loss: 1.111719
Train Epoch: 3 [17920/20000 (89%)]	Loss: 1.141732
Train Epoch: 3 [19200/20000 (96%)]	Loss: 0.935053

Train set: Accuracy: 70.62%


Validation set: Loss: 30.2154, Accuracy: 62.88%

Saved new best model with accuracy: 62.88%
Train Epoch: 4 [0/20000 (0%)]	Loss: 0.732595
Train Epoch: 4 [1280/20000 (6%)]	Loss: 0.706676
Train Epoch: 4 [2560/20000 (13%)]	Loss: 0.780640
Train Epoch: 4 [3840/20000 (19%)]	Loss: 0.731170
Train Epoch: 4 [5120/20000 (25%)]	Loss: 0.758329
Train Epoch: 4 [6400/20000 (32%)]	Loss: 0.632115
Train Epoch: 4 [7680/20000 (38%)]	Loss: 0.730809
Train Epoch: 4 [8960/20000 (45%)]	Loss: 0.724379
Train Epoch: 4 [10240/20000 (51%)]	Loss: 0.852299
Train Epoch: 4 [11520/20000 (57%)]	Loss: 0.733530
Train Epoch: 4 [12800/20000 (64%)]	Loss: 0.782642
Train Epoch: 4 [14080/20000 (70%)]	Loss: 0.782139
Train Epoch: 4 [15360/20000 (76%)]	Loss: 0.561011
Train Epoch: 4 [16640/20000 (83%)]	Loss: 0.580509
Train Epoch: 4 [17920/20000 (89%)]	Loss: 0.819080
Train Epoch: 4 [19200/20000 (96%)]	Loss: 0.652614

Train set: Accuracy: 80.62%


Validation set: Loss: 27.1983, Accuracy: 68.80%

Saved new best model with accuracy: 68.80%
Train Epoch: 5 [0/20000 (0%)]	Loss: 0.499505
Train Epoch: 5 [1280/20000 (6%)]	Loss: 0.556848
Train Epoch: 5 [2560/20000 (13%)]	Loss: 0.366506
Train Epoch: 5 [3840/20000 (19%)]	Loss: 0.359587
Train Epoch: 5 [5120/20000 (25%)]	Loss: 0.455072
Train Epoch: 5 [6400/20000 (32%)]	Loss: 0.399847
Train Epoch: 5 [7680/20000 (38%)]	Loss: 0.333763
Train Epoch: 5 [8960/20000 (45%)]	Loss: 0.521399
Train Epoch: 5 [10240/20000 (51%)]	Loss: 0.432774
Train Epoch: 5 [11520/20000 (57%)]	Loss: 0.276969
Train Epoch: 5 [12800/20000 (64%)]	Loss: 0.446566
Train Epoch: 5 [14080/20000 (70%)]	Loss: 0.325732
Train Epoch: 5 [15360/20000 (76%)]	Loss: 0.395874
Train Epoch: 5 [16640/20000 (83%)]	Loss: 0.457149
Train Epoch: 5 [17920/20000 (89%)]	Loss: 0.504920
Train Epoch: 5 [19200/20000 (96%)]	Loss: 0.331291

Train set: Accuracy: 88.05%


Validation set: Loss: 25.0578, Accuracy: 72.20%

Saved new best model with accuracy: 72.20%
Train Epoch: 6 [0/20000 (0%)]	Loss: 0.283848
Train Epoch: 6 [1280/20000 (6%)]	Loss: 0.386248
Train Epoch: 6 [2560/20000 (13%)]	Loss: 0.207017
Train Epoch: 6 [3840/20000 (19%)]	Loss: 0.181735
Train Epoch: 6 [5120/20000 (25%)]	Loss: 0.250777
Train Epoch: 6 [6400/20000 (32%)]	Loss: 0.163538
Train Epoch: 6 [7680/20000 (38%)]	Loss: 0.278701
Train Epoch: 6 [8960/20000 (45%)]	Loss: 0.184826
Train Epoch: 6 [10240/20000 (51%)]	Loss: 0.225560
Train Epoch: 6 [11520/20000 (57%)]	Loss: 0.213784
Train Epoch: 6 [12800/20000 (64%)]	Loss: 0.308050
Train Epoch: 6 [14080/20000 (70%)]	Loss: 0.475031
Train Epoch: 6 [15360/20000 (76%)]	Loss: 0.286812
Train Epoch: 6 [16640/20000 (83%)]	Loss: 0.297644
Train Epoch: 6 [17920/20000 (89%)]	Loss: 0.322583
Train Epoch: 6 [19200/20000 (96%)]	Loss: 0.270270

Train set: Accuracy: 92.85%


Validation set: Loss: 23.6187, Accuracy: 74.56%

Saved new best model with accuracy: 74.56%
Train Epoch: 7 [0/20000 (0%)]	Loss: 0.190865
Train Epoch: 7 [1280/20000 (6%)]	Loss: 0.155712
Train Epoch: 7 [2560/20000 (13%)]	Loss: 0.222097
Train Epoch: 7 [3840/20000 (19%)]	Loss: 0.239859
Train Epoch: 7 [5120/20000 (25%)]	Loss: 0.155875
Train Epoch: 7 [6400/20000 (32%)]	Loss: 0.150772
Train Epoch: 7 [7680/20000 (38%)]	Loss: 0.134851
Train Epoch: 7 [8960/20000 (45%)]	Loss: 0.166931
Train Epoch: 7 [10240/20000 (51%)]	Loss: 0.130413
Train Epoch: 7 [11520/20000 (57%)]	Loss: 0.121300
Train Epoch: 7 [12800/20000 (64%)]	Loss: 0.213457
Train Epoch: 7 [14080/20000 (70%)]	Loss: 0.207572
Train Epoch: 7 [15360/20000 (76%)]	Loss: 0.099308
Train Epoch: 7 [16640/20000 (83%)]	Loss: 0.205295
Train Epoch: 7 [17920/20000 (89%)]	Loss: 0.141645
Train Epoch: 7 [19200/20000 (96%)]	Loss: 0.291080

Train set: Accuracy: 95.78%


Validation set: Loss: 23.7068, Accuracy: 75.40%

Saved new best model with accuracy: 75.40%
Train Epoch: 8 [0/20000 (0%)]	Loss: 0.125415
Train Epoch: 8 [1280/20000 (6%)]	Loss: 0.143709
Train Epoch: 8 [2560/20000 (13%)]	Loss: 0.129063
Train Epoch: 8 [3840/20000 (19%)]	Loss: 0.161200
Train Epoch: 8 [5120/20000 (25%)]	Loss: 0.079722
Train Epoch: 8 [6400/20000 (32%)]	Loss: 0.086742
Train Epoch: 8 [7680/20000 (38%)]	Loss: 0.053579
Train Epoch: 8 [8960/20000 (45%)]	Loss: 0.135066
Train Epoch: 8 [10240/20000 (51%)]	Loss: 0.142540
Train Epoch: 8 [11520/20000 (57%)]	Loss: 0.067992
Train Epoch: 8 [12800/20000 (64%)]	Loss: 0.052966
Train Epoch: 8 [14080/20000 (70%)]	Loss: 0.118394
Train Epoch: 8 [15360/20000 (76%)]	Loss: 0.060454
Train Epoch: 8 [16640/20000 (83%)]	Loss: 0.152015
Train Epoch: 8 [17920/20000 (89%)]	Loss: 0.055512
Train Epoch: 8 [19200/20000 (96%)]	Loss: 0.168786

Train set: Accuracy: 97.19%


Validation set: Loss: 23.2048, Accuracy: 76.64%

Saved new best model with accuracy: 76.64%
Train Epoch: 9 [0/20000 (0%)]	Loss: 0.108997
Train Epoch: 9 [1280/20000 (6%)]	Loss: 0.108137
Train Epoch: 9 [2560/20000 (13%)]	Loss: 0.090471
Train Epoch: 9 [3840/20000 (19%)]	Loss: 0.082900
Train Epoch: 9 [5120/20000 (25%)]	Loss: 0.120256
Train Epoch: 9 [6400/20000 (32%)]	Loss: 0.041119
Train Epoch: 9 [7680/20000 (38%)]	Loss: 0.077339
Train Epoch: 9 [8960/20000 (45%)]	Loss: 0.056185
Train Epoch: 9 [10240/20000 (51%)]	Loss: 0.060084
Train Epoch: 9 [11520/20000 (57%)]	Loss: 0.066613
Train Epoch: 9 [12800/20000 (64%)]	Loss: 0.063232
Train Epoch: 9 [14080/20000 (70%)]	Loss: 0.178166
Train Epoch: 9 [15360/20000 (76%)]	Loss: 0.074429
Train Epoch: 9 [16640/20000 (83%)]	Loss: 0.163077
Train Epoch: 9 [17920/20000 (89%)]	Loss: 0.100643
Train Epoch: 9 [19200/20000 (96%)]	Loss: 0.134303

Train set: Accuracy: 97.86%


Validation set: Loss: 22.9005, Accuracy: 76.76%

Saved new best model with accuracy: 76.76%
Train Epoch: 10 [0/20000 (0%)]	Loss: 0.091276
Train Epoch: 10 [1280/20000 (6%)]	Loss: 0.029697
Train Epoch: 10 [2560/20000 (13%)]	Loss: 0.111859
Train Epoch: 10 [3840/20000 (19%)]	Loss: 0.055733
Train Epoch: 10 [5120/20000 (25%)]	Loss: 0.075300
Train Epoch: 10 [6400/20000 (32%)]	Loss: 0.093617
Train Epoch: 10 [7680/20000 (38%)]	Loss: 0.085468
Train Epoch: 10 [8960/20000 (45%)]	Loss: 0.093047
Train Epoch: 10 [10240/20000 (51%)]	Loss: 0.045046
Train Epoch: 10 [11520/20000 (57%)]	Loss: 0.063008
Train Epoch: 10 [12800/20000 (64%)]	Loss: 0.080231
Train Epoch: 10 [14080/20000 (70%)]	Loss: 0.078563
Train Epoch: 10 [15360/20000 (76%)]	Loss: 0.109231
Train Epoch: 10 [16640/20000 (83%)]	Loss: 0.109160
Train Epoch: 10 [17920/20000 (89%)]	Loss: 0.038781
Train Epoch: 10 [19200/20000 (96%)]	Loss: 0.112533

Train set: Accuracy: 98.27%


Validation set: Loss: 22.7757, Accuracy: 76.84%

Saved new best model with accuracy: 76.84%
Train Epoch: 11 [0/20000 (0%)]	Loss: 0.072011
Train Epoch: 11 [1280/20000 (6%)]	Loss: 0.211471
Train Epoch: 11 [2560/20000 (13%)]	Loss: 0.181308"""