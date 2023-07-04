# Used by "import_matrix"



class Pointer_like:
    def __init__(self, obj):
        self.obj = obj

    def get(self):
        return self.obj

    def set(self, obj):
        self.obj = obj

    # define the oeprators
    def __iadd__(self, other):
        self.obj = self.get() + other.get()
        return self

    def __add__(self, other):
        new_ptr = Pointer_like(self.get() + other.get())
        return new_ptr

    def __sub__(self, other):
        new_ptr = Pointer_like(self.get() - other.get())
        return new_ptr

    def __mul__(self, other):
        new_ptr = Pointer_like(self.get() * other.get())
        return new_ptr

    def __truediv__(self, other):
        new_ptr = Pointer_like(self.get() / other.get())
        return new_ptr

    # define comparisons
    def __lt__(self, other):
        return self.get() < other.get()

    def __le__(self, other):
        return self.get() <= other.get()

    def __eq__(self, other):
        return self.get() == other.get()

    def __neq__(self, other):
        return self.get() != other.get()

    def __ge__(self, other):
        return self.get() >= other.get()

    def __gt__(self, other):
        return self.get() > other.get()

    # define casting
    def __float__(self):
        return self.get()


if __name__ == '__main__':
    # demonstrating of desired behavior
    x_ptr = [Pointer_like(i) for i in range(0,10)]
    zero_ptr = Pointer_like(0)

    depending_array = [zero_ptr, zero_ptr, x_ptr[4], zero_ptr, x_ptr[9]]

    print([ele.get() for ele in depending_array])
    x_ptr[4].set(10)
    print([ele.get() for ele in depending_array])
