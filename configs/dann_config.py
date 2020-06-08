# features extraction
LOSS_NEED_INTERMEDIATE_LAYERS = False
# after_conv for features from the last convolution layer,
# before_class for features before classifier
# before_bottleneck for features before bottleneck
FEATURES = "after_conv"

UNK_VALUE = -100  # torch default
BACKBONE_PRETRAINED = True
FREEZE_BACKBONE_FEATURES = True
# possible for AlexNet: 0, 2, 4, 6, 8, 10
# possible for ResNet: 0, 1, 3, 33, 72, 129, 141, 159
FREEZE_LEVEL = 129
BATCH_SIZE = 64

LR = 0.01
NUM_WORKERS = 4
N_EPOCHS = 350
STEPS_PER_EPOCH = 30
VAL_FREQ = 3
SAVE_MODEL_FREQ = 349

IMAGE_SIZE = 224
DATASET = "office-31" # "office-31", 'visda'
CLASSES_CNT = 31 # 31 office, 12 visda
SOURCE_DOMAIN = "amazon"
TARGET_DOMAIN = "webcam"

################### Model dependent parameters #########################

# select backbone: alexnet resnet50 resnet50_rich vanilla_dann for DANN,
# DANN-CA_simple DANN-CA_rich for DANN-CA, DADA
MODEL_BACKBONE = "resnet50_rich" # alexnet resnet50 resnet50_rich vanilla_dann DANN-CA_simple DANN-CA_rich
DOMAIN_HEAD = "vanilla_dann"

# DANN
DOMAIN_LOSS = 1
CLASSIFICATION_LOSS = 2 - DOMAIN_LOSS
LOSS_GAMMA = 10  # from authors, not optimized
IS_UNSUPERVISED = True
GRADIENT_REVERSAL_LAYER_ALPHA = 1.0
NEED_ADAPTATION_BLOCK = False # ="True" only for alexnet, ="False" for other types
NEED_ADAPTATION_BLOCK_AV = False
BOTTLENECK_SIZE = 256
DROPOUT = 0.2
BLOCKS_WITH_SMALLER_LR = 0 # ="2" only for alexnet, ="0" for other types
ENTROPY_REG = True
ENTROPY_REG_COEF = 1

if NEED_ADAPTATION_BLOCK and NEED_ADAPTATION_BLOCK_AV:
    raise RuntimeError('select only one adaptation block type')


# DANN_CA & DADA
FEATURES_END = 159
ALTERNATING_UPDATE = False
LONG_CLS = True

# DANN_CA
DANN_CA = False

# DADA
DADA = False
NUM_EPOCH_PRETRAIN = 30


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
