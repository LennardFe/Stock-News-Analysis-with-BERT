from enum import StrEnum

class Target(StrEnum):
    D3 = "3D"
    W1 = "1W"
    W2 = "2W"
    M1 = "1M"
    M3 = "3M"

class TMode(StrEnum):
    STATIC = "STATIC"
    NORMAL_DISTRIBUTION = "NORMAL_DISTRIBUTION"
    PERCENTAGE = "PERCENTAGE"

class StockExchange(StrEnum):
    NYSE = "NYSE"
    LSE = "LSE"
    EUREX = "EUREX"