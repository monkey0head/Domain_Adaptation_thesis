LOSS_GAMMA = 10  # from authors, not optimized
LOSS_NEED_INTERMEDIATE_LAYERS = True
# after_conv for features from the last convolution layer,
# before_class for features before classifier
# before_bottleneck for features before bottleneck
FEATURES = "after_conv"
UNK_VALUE = -100  # torch default
IS_UNSUPERVISED = True

GRADIENT_REVERSAL_LAYER_ALPHA = 1.0
FREEZE_BACKBONE_FEATURES = True
# possible for AlexNet: 0, 2, 4, 6, 8, 10
# possible for ResNet: 0, 1, 3, 33, 72, 129, 141, 159
FREEZE_LEVEL = 141
BATCH_SIZE = 64

LR = 0.01
NUM_WORKERS = 4
N_EPOCHS = 200
STEPS_PER_EPOCH = 10
VAL_FREQ = 1
SAVE_MODEL_FREQ = 199

################### Model dependent parameters #########################

CLASSES_CNT = 31
MODEL_BACKBONE = "DANN-CA_rich" # alexnet resnet50 resnet50_rich vanilla_dann DANN-CA_simple DANN-CA_rich
# MODEL_BACKBONE = "resnet50_rich" # alexnet resnet50 resnet50_rich vanilla_dann DANN-CA_simple DANN-CA_rich
# BOTTLENECK_SIZE = 256
DOMAIN_HEAD = "vanilla_dann"
DOMAIN_LOSS = 1
CLASSIFICATION_LOSS = 2 - DOMAIN_LOSS
BACKBONE_PRETRAINED = True
NEED_ADAPTATION_BLOCK = False # ="True" only for alexnet, ="False" for other types
# NEED_ADAPTATION_BLOCK_AV = False
BLOCKS_WITH_SMALLER_LR = 0 # ="2" only for alexnet, ="0" for other types
IMAGE_SIZE = 224
DATASET = "office-31"
SOURCE_DOMAIN = "amazon"
TARGET_DOMAIN = "webcam"
ENTROPY_REG = True
ENTROPY_REG_COEF = 0.2


# CLASSES_CNT = 10
# MODEL_BACKBONE = "mnist_dann"
# DOMAIN_HEAD = "mnist_dann"
# BACKBONE_PRETRAINED = False
# NEED_ADAPTATION_BLOCK = False
# BLOCKS_WITH_SMALLER_LR = 0
# IMAGE_SIZE = 28
# DATASET = "mnist"
# SOURCE_DOMAIN = "mnist"
# TARGET_DOMAIN = "mnist-m"

# DANN_CA
DANN_CA = True
LAMBDA = 1
FEATURES_END = 159
ALTERNATING_UPDATE = False
LONG_CLS = True
NEED_ADAPTATION_BLOCK_AV = False
BOTTLENECK_SIZE = 256
DROPOUT = 0.2


if NEED_ADAPTATION_BLOCK and NEED_ADAPTATION_BLOCK_AV:
    raise RuntimeError('select only one adaptation block type')