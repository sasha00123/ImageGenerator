import os
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from random import choice, randint, shuffle
from typing import List

from tqdm import trange
from PIL import Image
import numpy as np
from skimage.color import rgb2lab
from colorthief import ColorThief

from config import *

images = []


@dataclass
class Chromosome:
    # Genes
    x: int = field(default_factory=lambda: randint(1, CANVAS_WIDTH))
    y: int = field(default_factory=lambda: randint(1, CANVAS_HEIGHT))

    # Future improvements
    # w: int = field(default_factory=lambda: randint(1, CANVAS_WIDTH))
    # h: int = field(default_factory=lambda: randint(1, CANVAS_HEIGHT))
    # rotation: int = field(default_factory=lambda: randint(-360, 360))

    image: str = field(default_factory=lambda: choice(images))


@dataclass
class Hypothesis:
    chromosomes: List[Chromosome] = field(default_factory=lambda: [Chromosome() for _ in range(16)])


@dataclass
class Experiment:
    population: List[Hypothesis] = field(default_factory=lambda: [Hypothesis() for _ in range(POPULATION_SIZE)])


def fitness_function(img1: Image):
    img1 = np.array(rgb2lab(img1.convert('RGB')))
    with Image.open(TARGET_IMAGE) as target:
        target = np.array(rgb2lab(target.convert('RGB')))
        return np.linalg.norm(img1 - target)


def breed(hyp1: Hypothesis, hyp2: Hypothesis):
    new_chromosomes = []

    old_chromosomes = hyp1.chromosomes + hyp2.chromosomes
    shuffle(old_chromosomes)

    for chromosome in old_chromosomes[:HYPOTHESIS_SIZE]:
        if randint(1, 100) % 100 <= CHANCE_TAKE_MODIFIED_CHROMOSOME:
            new_chromosomes.append(Chromosome())
        else:
            new_chromosomes.append(chromosome)

    return Hypothesis(new_chromosomes)


def make_image(hyp: Hypothesis):
    canvas = Image.new('RGBA', (CANVAS_WIDTH, CANVAS_HEIGHT))
    for chromosome in hyp.chromosomes:
        canvas.paste(chromosome.image, (chromosome.x, chromosome.y), chromosome.image)
    return canvas


def loss(x):
    return fitness_function(make_image(x))


def next_generation(exp: Experiment):
    exp = deepcopy(exp)
    shuffle(exp.population)

    for _ in range(NUM_BREEDS_PER_GENERATION):
        hyp1 = choice(exp.population)
        hyp2 = choice(exp.population)

        exp.population.append(breed(hyp1, hyp2))

    exp.population = list(sorted(exp.population, key=loss))[:POPULATION_SIZE]
    return exp


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 main.py <source> <target>")

    color_thief = ColorThief(TARGET_IMAGE)
    palette = color_thief.get_palette(color_count=10)

    for color in palette:
        for image_path in os.listdir(BLOTS_DIR):
            with Image.open(os.path.join(BLOTS_DIR, image_path)) as img:
                img = img.convert('RGBA')
                pixels = img.load()
                for x in range(img.width):
                    for y in range(img.height):
                        r, g, b, a = pixels[x, y]
                        if (r, g, b) == (26, 26, 26):
                            pixels[x, y] = (*color, a)
                img.thumbnail((48, 48))
                images.append(img)

    exp = Experiment()
    for i in trange(NUM_GENERATIONS):
        exp = next_generation(exp)
        best = min(exp.population, key=lambda x: fitness_function(make_image(x)))
        print(fitness_function(make_image(best)))
        make_image(best).save(CHECKPOINT_IMAGE)
