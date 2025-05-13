from enum import StrEnum, auto


class Category(StrEnum):
    Geo = "Geological"
    Met = "Meteorological"
    Safety = auto()
    Security = auto()
    Rescue = auto()
    Fire = auto()
    Health = auto()
    Env = "Environmental"
    Transport = auto()
    Infra = "Infrastructure"
    Other = auto()


class Severity(StrEnum):
    Extreme = auto()
    Severe = auto()
    Moderate = auto()
    Minor = auto()


class Urgency(StrEnum):
    Immediate = auto()
    Expected = auto()
    Future = auto()
    Past = auto()


class Certainty(StrEnum):
    VeryLikely = "Very Likely"
    Likely = auto()
    Possible = auto()
    Unlikely = auto()


class Status(StrEnum):
    Actual = auto()
    Exercise = auto()
    System = auto()
    Test = auto()


class MsgType(StrEnum):
    Alert = auto()
    Update = auto()
    Cancel = auto()


class Scope(StrEnum):
    Public = auto()
    Restricted = auto()
    Private = auto()
