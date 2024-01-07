from typing import Literal, TypedDict


class PatchDataset(TypedDict):
    name: str
    path: str


type PatchNormalization = Literal["local"] | Literal["global"] | Literal["both"]
