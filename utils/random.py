"""
Controlled randomness and config parsing

Created on Thu Oct 12 12:51:39 2023

@author: jack
"""

from random import Random
from typing import Optional
import math


def log10_uniform(
    min_val: float,
    max_val: float,
    dps: Optional[int] = None,
    generator: Optional[Random] = None,
) -> float:

    # -- get min and max values in log10 space
    log_min = math.log10(min_val)
    log_max = math.log10(max_val)

    # -- generate uniform number in log space
    log_uniform_value = uniform(log_min, log_max, dps, generator=generator)

    # -- convert back to original space
    val = 10**log_uniform_value

    # -- round if required
    if dps is not None:
        val = round(val, dps)

    return val


def randpower(min_val, max_val, power, n=1, repeats=None, generator=None):
    print(generator)
    num = randint(min_val=min_val, max_val=max_val, n=n, generator=generator)

    if n == 1:
        if repeats is None or repeats < 2:
            return power**num
        else:
            num = power**num
            return [num for _ in range(repeats)]
    else:
        return [power**i for i in num]


def uniform(
    min_val: float,
    max_val: float,
    dps: Optional[int] = None,
    generator: Optional[Random] = None,
) -> float:

    generator = generator or Random()

    val = generator.uniform(min_val, max_val)

    if dps:
        val = round(val, dps)

    return val


def randint(
    min_val: int, max_val: int, n=1, generator: Optional[Random] = None
) -> float:
    generator = generator or Random()

    if n == 1:
        return generator.randint(min_val, max_val)
    else:
        return [randint(min_val, max_val, 1, generator) for _ in range(n)]


def randchoice(*args: list, generator: Optional[Random] = None):
    return args[randint(0, len(args) - 1, generator=generator)]
