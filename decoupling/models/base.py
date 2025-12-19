from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    """
    Abstract base class for clustering models.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any = None) -> "BaseModel":
        """
        Fit the model.
        """
        pass

    @abstractmethod
    def fit_predict(self, *args, **kwargs):
        """
        Fit the model and return cluster labels.
        """
        pass
