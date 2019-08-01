class MyCircularQueue:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the queue to be k.
        """
        self.size = k
        self.front_pointer = 0
        self.end_pointer = 0
        self.queue = [None] * k
        self.isfull = False
        self.isempty = True

    def enQueue(self, value: int) -> bool:
        """
        Insert an element into the circular queue. Return true if the operation is successful.
        """
        self.isempty = False
        if not self.isFull():
            self.queue[self.end_pointer] = value
            self.end_pointer = self._next_pos(self.end_pointer)
            if self.end_pointer == self.front_pointer:
                self.isfull = True
            return True
        else:
            return False

    def deQueue(self) -> bool:
        """
        Delete an element from the circular queue. Return true if the operation is successful.
        """
        self.isfull = False
        if not self.isEmpty():
            self.front_pointer = self._next_pos(self.front_pointer)
            if self.front_pointer == self.end_pointer:
                self.isempty = True
            return True
        else:
            return False

    def Front(self) -> int:
        """
        Get the front item from the queue.
        """
        if not self.isEmpty():
            return self.queue[self.front_pointer]
        else:
            return -1

    def Rear(self) -> int:
        """
        Get the last item from the queue.
        """
        if not self.isEmpty():
            return self.queue[self.prev_pos(self.end_pointer)]
        else:
            return -1

    def isEmpty(self) -> bool:
        """
        Checks whether the circular queue is empty or not.
        """
        return self.isempty

    def isFull(self) -> bool:
        """
        Checks whether the circular queue is full or not.
        """
        return self.isfull

    def _next_pos(self, loc: int) -> int:
        if loc == self.size - 1:
            loc = 0
        else:
            loc += 1
        return loc

    def _prev_pos(self, loc: int) -> int:
        if loc == 0:
            loc = self.size - 1
        else:
            loc -= 1
        return loc

# Your MyCircularQueue object will be instantiated and called as such:
obj = MyCircularQueue(3)
param_1 = obj.enQueue(1)
param_2 = obj.enQueue(2)
param_3 = obj.enQueue(3)
param_4 = obj.enQueue(4)
param_5 = obj.deQueue()
param_6 = obj.Front()
param_4 = obj.Rear()
param_5 = obj.isEmpty()
param_6 = obj.isFull()