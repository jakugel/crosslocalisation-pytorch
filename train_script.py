import torch
from data_loaders import get_data_labelled, get_data_unlabelled
from generator_network import Generator
from discriminator_networks import DiscriminatorLabelled, DiscriminatorUnlabelled
from mapping_networks import MappingNetworkLabelled, MappingNetworkUnlabelled

from generator_network_custom import GeneratorCustom
from discriminator_networks_custom import DiscriminatorLabelledCustom, DiscriminatorUnlabelledCustom
from mapping_networks_custom import MappingNetworkLabelledCustom, MappingNetworkUnlabelledCustom

from penalties import PathLengthPenalty
from torch import optim
from training import train_fn
from math import log2

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
LR = 5e-5
ADAM_B1 = 0.0
ADAM_B2 = 0.999
NUM_CLASSES = 10
MIXING_PROB = 0.9
BATCH_SIZE = 64
LAMBDA_GP = 10
PL_BETA = 0.99
IMG_SIZE = 32
LOG_RESOLUTION = int(log2(IMG_SIZE))
NUM_EPOCHS = 100000
H5FILEPATH = "./data/your_data.hdf5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOSS_FN = "hinge"    # "wgan", "hinge"
EQUALIZED = False    # whether to use equalized layers or not
CROSS_LEVEL_TRAIN = 1
CROSS_LEVEL_TEST = NUM_GEN_LAYERS - 1
USE_CUSTOM_NETS = True
SAVE_MODELS = True
SAVE_MODELS_FREQ = 1000
SAVE_MODELS_DIR = "./crossloc_models"
SAVE_IMAGES = True
SAVE_IMAGES_FREQ = 100
SAVE_IMAGES_DIR = "./crossloc_images"

# "lo": labelled only training, "uo": unlabelled only training, "mixed": mixed training,
# "cross": mixed training with cross-localisation
TRAINING_MODE = "cross"

# setup networks
if USE_CUSTOM_NETS:
    gen = GeneratorCustom(LOG_RESOLUTION, W_DIM, START_GEN_FEATURES, NUM_GEN_LAYERS, EQUALIZED).to(DEVICE)
else:
    gen = Generator(LOG_RESOLUTION, W_DIM, START_GEN_FEATURES, MAX_GEN_FEATURES, NUM_GEN_LAYERS, EQUALIZED).to(DEVICE)

if TRAINING_MODE in ["lo", "mixed", "cross"]:
    images_l, labels = get_data_labelled(H5FILEPATH, image_size=IMG_SIZE)
    if USE_CUSTOM_NETS:
        criticl = DiscriminatorLabelledCustom(LOG_RESOLUTION, START_DIS_FEATURES, NUM_CLASSES, NUM_DIS_LAYERS,
                                        MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
        mapnetl = MappingNetworkLabelledCustom(Z_DIM, W_DIM, NUM_MAP_LAYERS, NUM_CLASSES, EQUALIZED, MAP_LEAKY).to(DEVICE)
    else:
        criticl = DiscriminatorLabelled(LOG_RESOLUTION, START_DIS_FEATURES, MAX_DIS_FEATURES, NUM_CLASSES, NUM_DIS_LAYERS,
                                        MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
        mapnetl = MappingNetworkLabelled(Z_DIM, W_DIM, NUM_MAP_LAYERS, NUM_CLASSES, EQUALIZED, MAP_LEAKY).to(DEVICE)

    opt_criticl = optim.Adam(criticl.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2))
    opt_mapnetl = optim.Adam(mapnetl.parameters(), lr=LR * MAPPING_LR_MULT, betas=(ADAM_B1, ADAM_B2))

    criticl.train()
    mapnetl.train()
else:
    criticl = None
    mapnetl = None
    opt_criticl = None
    opt_mapnetl = None
    images_l, labels = None, None

if TRAINING_MODE in ["uo", "mixed", "cross"]:
    images_u = get_data_unlabelled(H5FILEPATH, image_size=IMG_SIZE)

    if USE_CUSTOM_NETS:
        criticu = DiscriminatorUnlabelledCustom(LOG_RESOLUTION, START_DIS_FEATURES, NUM_DIS_LAYERS,
                                          MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
        mapnetu = MappingNetworkUnlabelledCustom(Z_DIM, W_DIM, NUM_MAP_LAYERS, EQUALIZED, MAP_LEAKY).to(DEVICE)
    else:
        criticu = DiscriminatorUnlabelled(LOG_RESOLUTION, START_DIS_FEATURES, MAX_DIS_FEATURES, NUM_DIS_LAYERS,
                                          MDL_KERNELS, MDL_KERNEL_SIZE, EQUALIZED, DIS_DROPOUT).to(DEVICE)
        mapnetu = MappingNetworkUnlabelled(Z_DIM, W_DIM, NUM_MAP_LAYERS, EQUALIZED, MAP_LEAKY).to(DEVICE)

    opt_criticu = optim.Adam(criticu.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2))
    opt_mapnetu = optim.Adam(mapnetu.parameters(), lr=LR * MAPPING_LR_MULT, betas=(ADAM_B1, ADAM_B2))

    criticu.train()
    mapnetu.train()
else:
    criticu = None
    mapnetu = None
    opt_criticu = None
    opt_mapnetu = None
    images_u = None

path_length_penalty = PathLengthPenalty(PL_BETA).to(DEVICE)
opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(ADAM_B1, ADAM_B2))
gen.train()

train_fn(
    criticl, criticu,
    gen, mapnetl, mapnetu,
    path_length_penalty,
    images_l, labels, images_u,
    opt_criticl, opt_criticu,
    opt_gen,
    opt_mapnetl, opt_mapnetu, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=DEVICE,
    num_classes=NUM_CLASSES, img_size=IMG_SIZE, w_dim=W_DIM,
    mixing_prob=MIXING_PROB, logres=LOG_RESOLUTION, lambda_gp=LAMBDA_GP, num_gen_layers=NUM_GEN_LAYERS,
    training_mode=TRAINING_MODE, loss=LOSS_FN, cross_level_train=CROSS_LEVEL_TRAIN, cross_level_test=CROSS_LEVEL_TEST,
    use_custom_nets=USE_CUSTOM_NETS, save_models=SAVE_MODELS, save_models_freq=SAVE_MODELS_FREQ,
    save_models_dir=SAVE_MODELS_DIR, save_images=SAVE_IMAGES, save_images_freq=SAVE_IMAGES_FREQ,
    save_images_dir=SAVE_IMAGES_DIR
)