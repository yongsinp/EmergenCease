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
    Actual = auto()  # Actionable by all targeted recipients
    Exercise = auto()  # Actionable only by designated exercise participants
    System = auto()  # For messages that support alert network internal functions
    Test = auto()


class MsgType(StrEnum):
    Alert = auto()  # Initial information requiring attention by targeted recipients
    Update = auto()  # Updates and supersedes the earlier message(s)
    Cancel = auto()  # Cancels the earlier message(s)
    Ack = "Acknowledgment"  # Acknowledges receipt and acceptance of the message(s)


class Scope(StrEnum):
    Public = auto()  # For general dissemination to unrestricted audiences.
    Restricted = auto()
    Private = auto()
