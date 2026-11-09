class MinHeap:


    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def push(self, value):
        self.heap.append(value)
        self._sift_up(len(self.heap) - 1)

    def peek(self):
        if not self.heap:
            return None

        return self.heap[0]

    def pop(self):
        if not self.heap:
            return None

        minimum = self.heap[0]
        last = self.heap.pop()

        if self.heap:
            self.heap[0] = last
            self._sift_down(0)

        return minimum

    def _sift_up(self, index):
        while index > 0:
            parent = (index - 1) // 2

            if self.heap[index] >= self.heap[parent]:
                break

            self.heap[index], self.heap[parent] = (
                self.heap[parent],
                self.heap[index]
            )

            index = parent

    def _sift_down(self, index):
        size = len(self.heap)

        while True:
            smallest = index
            left = 2 * index + 1
            right = 2 * index + 2

            if left < size and self.heap[left] < self.heap[smallest]:
                smallest = left

            if right < size and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest == index:
                break

            self.heap[index], self.heap[smallest] = (
                self.heap[smallest],
                self.heap[index]
            )

            index = smallest


def get_top_selling_products(df, limit=10):

    products = (
        df.groupby('ProductID')
        .agg(
            ProductName=('ProductName', 'first'),
            total_quantity=('Quantity', 'sum')
        )
        .reset_index()
    )

    heap = MinHeap()

    for row in products.itertuples(index=False):
        item = (
            int(row.total_quantity),
            str(row.ProductID),
            str(row.ProductName)
        )

        heap.push(item)

        if len(heap) > limit:
            heap.pop()

    result = []

    while len(heap) > 0:
        quantity, product_id, product_name = heap.pop()

        result.append({
            'ProductID': product_id,
            'ProductName': product_name,
            'total_quantity': quantity
        })

    result.reverse()

    return result