from strenum import StrEnum
from enum import auto
from ..._utils import BaseEnumMeta

class ContextLogitMode(StrEnum, metaclass=BaseEnumMeta):
    LAST_LOGIT = auto()
    ALL_LOGITS = auto()
    NONE_LOGITS = auto()