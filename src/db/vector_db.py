from abc import ABC, abstractmethod


class VectorDB(ABC):

    @abstractmethod
    def create_table(self):
        pass

    @abstractmethod
    def insert_documents(self, data: list[tuple]):
        pass

    @abstractmethod
    def search(self, query_vector: list[float], limit: int) -> list[tuple]:
        """
        - Row[0]: id
        - Row[1]: content
        - Row[2]: metadata
        - Row[3]: similarity
        """
        pass

    @abstractmethod
    def delete_all_documents(self):
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def close(self):
        pass
