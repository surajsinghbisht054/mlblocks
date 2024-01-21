from abc import ABC, abstractmethod

class AbstractDataHolder(ABC):
    
    @abstractmethod
    def get_dataset(self, index):
        pass

    @abstractmethod
    def get_test_dataset(self, index):
        pass

    