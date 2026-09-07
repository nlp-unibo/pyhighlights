from typing import Dict, List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_method

from pyhighlights.components.datasets import R2A_TASKS, R2A_URL

NAMESPACE = "pyhighlights"

R2A = RegistrationKey(name="dataset", tags={"r2a"}, namespace=NAMESPACE)


class R2AConfig(Configuration):
    """Beer and Hotel aspects; one variant per task."""

    # Cinnamon rejects a default that also appears in the variant list.
    task: str = Param(
        "hotel_Location",
        variants=[task for task in R2A_TASKS if task != "hotel_Location"],
    )
    splits: Dict[str, str] | None = Param(None)
    url: str = Param(R2A_URL)
    sha256: str | None = Param(None)
    directory: str | None = Param(None)
    remove_leakage: bool = Param(True)

    @classmethod
    @register_method(
        name="dataset",
        tags={"r2a"},
        namespace=NAMESPACE,
        component="pyhighlights.components.datasets.R2ALoader",
    )
    def default(cls):
        return super().default()


__all__: List[str] = ["R2A", "R2AConfig"]
