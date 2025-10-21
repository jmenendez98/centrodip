import argparse
import concurrent.futures
import os

import numpy as np
import scipy


class DipFilter:
    def __init__(
        self,
        min_size: int = 50,

    ):
        self.min_size = min_size