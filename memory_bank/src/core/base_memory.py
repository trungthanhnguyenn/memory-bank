from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

class BaseMemory(ABC):
    @abstractmethod
    def store(self, key: str, data: Any, metadata: Dict = None) -> bool:
        """
        Store data in the memory.

        Args:
            key (str): The key to identify the data.
            data (Any): The data to be stored.
            metadata (Dict, optional): Additional metadata for the data. Defaults to None.

        Returns:
            bool: True if the data was successfully stored, False otherwise.
        """
        pass
    @abstractmethod
    def retrieve(self, key: str) -> Optional[Any]:
        """
        Retrieve data from the memory.

        Args:
            key (str): The key to identify the data.

        Returns:
            Optional[Any]: The retrieved data, or None if the key does not exist.
        """
        pass
    @abstractmethod
    def search(self, query: str, threshold: float = 0.8) -> List[Tuple[str, Any, float]]:
        """
        Search for data in the memory based on a query.

        Args:
            query (str): The query to search for.
            threshold (float, optional): The similarity threshold for the search. Defaults to 0.8.

        Returns:
            List[Tuple[str, Any, float]]: A list of tuples containing the key, data, and similarity score of the matching items.
        """
        pass    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete data from the memory.

        Args:
            key (str): The key to identify the data.

        Returns:
            bool: True if the data was successfully deleted, False otherwise.
        """
        pass
    @abstractmethod
    def clear(self) -> int:
        """
        Clear all data from the memory.

        Returns:
            int: The number of items cleared.
        """
        pass        
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory.

        Returns:
            Dict[str, Any]: A dictionary containing memory statistics.
        """
        pass