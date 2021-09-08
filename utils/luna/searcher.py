class Searcher:
    def search(self, element):
        raise NotImplementedError
        
    def batch_search(self, elements):
        return [self.search(ele) for ele in elements]