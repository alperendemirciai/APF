class BaseSimilarity:
    """
    Base class for similarity measures.
    """
    
    def __init__(self):
        pass
        
    def compute_similarity(self, idx1: int, idx2: int) -> float:
        raise NotImplementedError("This method should be overridden by subclasses.")