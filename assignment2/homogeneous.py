import numpy as np

class HomogeneousArray(np.ndarray):
    def __new__(cls, input_array: list[int | float] | np.ndarray[int | float], homo: bool = False):
        input_array = np.asarray(input_array, dtype=float)

        assert input_array.ndim == 1, "HomogeneousArray must be 1D"

        obj = np.asarray([*input_array, 1], dtype=float)[..., np.newaxis].view(cls)

        if homo:
            obj = np.asarray(input_array, dtype=float)[..., np.newaxis].view(cls)
        return obj

    @property
    def dh(self):
        if self.size == 1:
            return self.flatten()[0]
        
        return np.array((self[:-1] / self[-1]).flatten())
    
    def __matmul__(self, other):
        return HomogeneousMatrix(np.matmul(self, other))
    

class HomogeneousMatrix(np.ndarray):
    def __new__(cls, input_array):
        input_array = np.asarray(input_array)
        if input_array.size == 1:
            return input_array.item()
        obj = np.asarray(input_array, dtype=float).view(cls)
        return obj

    @property
    def dh(self):
        return np.array(self / self.flatten()[-1])
    
    def __matmul__(self, other):
        return HomogeneousMatrix(np.matmul(self, other))
    

# Tests
assert np.all((HomogeneousArray([1, 2, 3]) * 3).dh == [1, 2, 3]) and np.all(HomogeneousArray([1, 2, 3]) * 3 == [[3], [6], [9], [3]])
assert HomogeneousArray([1, 2, 3]).T @ HomogeneousArray([1, 2, 3]) == 15
assert isinstance(HomogeneousArray([1, 2, 3]) @ HomogeneousArray([1, 2, 3]).T, HomogeneousMatrix)