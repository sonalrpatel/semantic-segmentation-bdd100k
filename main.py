import os
from sys import exit
from datetime import datetime
from pathlib import Path as path

from train import train
from test import test


if __name__ == '__main__':
    start = datetime.now()

    print(datetime.now() - start)
