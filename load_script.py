import torch
from generator_network import Generator
from discriminator_networks import DiscriminatorLabelled, DiscriminatorUnlabelled
from mapping_networks import MappingNetworkLabelled, MappingNetworkUnlabelled

from generator_network_custom import GeneratorCustom
from discriminator_networks_custom import DiscriminatorLabelledCustom, DiscriminatorUnlabelledCustom
from mapping_networks_custom import MappingNetworkLabelledCustom, MappingNetworkUnlabelledCustom

from synthesise_data import generate_examples
from math import log2

### LOAD TRAINED MODELS AND GENERATE DATA ###

# Generator parameters
NUM_GEN_LAYERS = 4
START_GEN_FEATURES = 32
MAX_GEN_FEATURES = 256

# Discriminator parameters
NUM_DIS_LAYERS = 4
START_DIS_FEATURES = 64
MAX_DIS_FEATURES = 256
DIS_DROPOUT = 0.25

# Minibatch discrimination parameters
MDL_KERNELS = 25
MDL_KERNEL_SIZE = 15

# Mapping network parameters
NUM_MAP_LAYERS = 2
MAPPING_LR_MULT = 0.1
W_DIM = 128
Z_DIM = 128
MAP_LEAKY = 0.2

# Other parameters
NUM_CLASSES = 10
IMG_SIZE = 32
LOG_RESOLUTION = int(log2(IMG_SIZE))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_FN = "hinge"    # "wgan", "hinge"
EQUALIZED = False    # whether to use equalized layers or not
CROSS_LEVEL_TEST = NUM_GEN_LAYERS - 1
USE_CUSTOM_NETS = True
SAVE_MODELS_DIR = "./crossloc_models"
MODEL_EPOCH = 2000
SAVE_IMAGES_DIR = "./crossloc_images_test"

# "lo": labelled only training, "uo": unlabelled only training, "mixed": mixed training,
# "cross": mixed training with cross-localisation
MODE = "cross"

# setup networks
if USE_CUSTOM_NETS:
    gen = GeneratorCustom(LOG_RESOLUTION, W_DIM, START_GEN_FEATURES, NUM_GEN_LAYERS, EQUALIZED).to(DEVICE)
else:
    gen = Generator(LOG_RESOLUTION, W_DIM, START_GEN_FEATURES, MAX_GEN_FEATURES, NUM_GEN_LAYERS, EQUALIZED).to(DEVICE)

gen.load_state_dict(
        torch.load(SAVE_MODELS_DIR + "/gen_" + str(MODEL_EPOCH) + ".pth"))
gen.eval()

if MODE in ["lo", "mixed", "cross"]:
    if USE_CUSTOM_NETS:
        criticl = DiscriminatorLabelledCustom(LOG_RESOLUTION, START_DIS_FEATURES, NUM_CLASSES, NUM_DIS_LAYERS,
                                        MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
        mapnetl = MappingNetworkLabelledCustom(Z_DIM, W_DIM, NUM_MAP_LAYERS, NUM_CLASSES, EQUALIZED, MAP_LEAKY).to(DEVICE)
    else:
        criticl = DiscriminatorLabelled(LOG_RESOLUTION, START_DIS_FEATURES, MAX_DIS_FEATURES, NUM_CLASSES, NUM_DIS_LAYERS,
                                        MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
        mapnetl = MappingNetworkLabelled(Z_DIM, W_DIM, NUM_MAP_LAYERS, NUM_CLASSES, EQUALIZED, MAP_LEAKY).to(DEVICE)

    criticl.load_state_dict(
        torch.load(SAVE_MODELS_DIR + "/criticl_" + str(MODEL_EPOCH) + ".pth"))
    mapnetl.load_state_dict(
        torch.load(SAVE_MODELS_DIR + "/mapnetl_" + str(MODEL_EPOCH) + ".pth"))

    criticl.eval()
    mapnetl.eval()
else:
    criticl = None
    mapnetl = None

if MODE in ["uo", "mixed", "cross"]:
    if USE_CUSTOM_NETS:
        criticu = DiscriminatorUnlabelledCustom(LOG_RESOLUTION, START_DIS_FEATURES, NUM_DIS_LAYERS,
                                          MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
        mapnetu = MappingNetworkUnlabelledCustom(Z_DIM, W_DIM, NUM_MAP_LAYERS, EQUALIZED, MAP_LEAKY).to(DEVICE)
    else:
        criticu = DiscriminatorUnlabelled(LOG_RESOLUTION, START_DIS_FEATURES, MAX_DIS_FEATURES, NUM_DIS_LAYERS,
                                          MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
        mapnetu = MappingNetworkUnlabelled(Z_DIM, W_DIM, NUM_MAP_LAYERS, EQUALIZED, MAP_LEAKY).to(DEVICE)

    criticu.load_state_dict(
        torch.load(SAVE_MODELS_DIR + "/criticu_" + str(MODEL_EPOCH) + ".pth"))
    mapnetu.load_state_dict(
        torch.load(SAVE_MODELS_DIR + "/mapnetu_" + str(MODEL_EPOCH) + ".pth"))

    criticu.eval()
    mapnetu.eval()
else:
    criticu = None
    mapnetu = None

generate_examples(gen, MODEL_EPOCH, CROSS_LEVEL_TEST, LOG_RESOLUTION, NUM_GEN_LAYERS, mapnetl, mapnetu,
                  W_DIM, num_classes=NUM_CLASSES, device=DEVICE, save_path=SAVE_IMAGES_DIR, training_mode=MODE,
                  use_custom_nets=USE_CUSTOM_NETS)