import os
from enum import Enum

from dotenv import load_dotenv


class Env(Enum):
    TEST = "TEST"
    PROD = "PROD"


def get_setting(key: str):
    load_dotenv()
    return os.getenv(key)
