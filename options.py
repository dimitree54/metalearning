# Data info
TARGET_WEIGHTS_GENERATION_BATCH_SIZE = 30000  # using max possible for speed up. 30000 works fine with 2Gb
TARGET_WEIGHTS_GENERATION_N_EPOCHS = 100

TN_DATA_GENERATION_BATCH_SIZE = 50  # using max possible for speed up. 50 works fine with 2Gb
TN_DATA_GENERATION_RESET_STEPS = 500  # 500  # this values chosen so that filling dataset takes 3 hours and requires
TN_DATA_GENERATION_DROP_RATE = 0.999997  # 1000 times to reset sn_weights
TN_DATA_SIZE = 1000000

SN_INPUT_SIZE = 784
SN_OUTPUT_SIZE = 10
SN_BATCH_SIZE = 1

LOCAL_INFO_SIZE = 3
GLOBAL_INFO_SIZE = 1
TN_INPUT_SIZE = LOCAL_INFO_SIZE + GLOBAL_INFO_SIZE
TN_OUTPUT_SIZE = 1
TN_BATCH_SIZE = 450000  # 450000

# SN description
SN_DESCRIPTION = [SN_INPUT_SIZE, 512, 256, 128, 64, 32, SN_OUTPUT_SIZE]  # we have fixed architecture for simplicity.
# In case of success we can try random architectures. Such architecture chosen because of two reasons: to look like
# encoder and to make first layer and all other have almost equal number of parameters.
WEIGHTS_SIGMA = 0.0001

# TN description
TN_DESCRIPTIONS = {
    "deep_nn": [TN_INPUT_SIZE, 8, 16, 32, 64, 128, 64, 32, 16, 8, TN_OUTPUT_SIZE],
    "nn": [TN_INPUT_SIZE, 64, 64, 64, TN_OUTPUT_SIZE],
    "linear": [TN_INPUT_SIZE, TN_OUTPUT_SIZE]}

# training parameters
TN_TRAINING_N_EPOCHS = 10000  # 10000
TN_N_TESTS = 100  # 100
TN_TEST_BS = 2  # 2
