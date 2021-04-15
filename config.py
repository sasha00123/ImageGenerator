import sys

BLOTS_DIR = 'blots'
TARGET_IMAGE = sys.argv[1]
CHECKPOINT_IMAGE = sys.argv[2]

CANVAS_WIDTH = 512  # px
CANVAS_HEIGHT = 512  # px

CHANCE_TAKE_MODIFIED_CHROMOSOME = 10  # %

HYPOTHESIS_SIZE = 1024
POPULATION_SIZE = 8
NUM_BREEDS_PER_GENERATION = 128
NUM_GENERATIONS = 1000
