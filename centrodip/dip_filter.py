import argparse
import concurrent.futures
import os

import numpy as np
import scipy


class DipFilter:
    def __init__(
        self,
        dips
    ):
        self.dips = dips