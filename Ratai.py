from collections import deque
import heapq

class WaterNetwork:
    def __init__(self, junctions):
        self.graph = {i: [] for i in range(junctions)}

    def add_connection(self, a, b, cost):
        # Since the network is bidirectional
        self.graph[a].append((b, cost))
        self.graph[b].append((a, cost))

# ---------------- Search Algorithms ---------------- #

def bfs_search(network, start, target):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        node, path = queue.popleft()
        if node == target:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor, _ in network.graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return None

def dfs_search(network, node, target, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(node)
    path.append(node)

    if node == target:
        return path

    for neighbor, _ in network.graph[node]:
        if neighbor not in visited:
            result = dfs_search(network, neighbor, target, visited, path)
            if result:
                return result

    path.pop()
    return None

def depth_limited_search(network, node, target, depth, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(node)
    path.append(node)

    if node == target:
        return path
    if depth == 0:
        visited.remove(node)
        path.pop()
        return None

    for neighbor, _ in network.graph[node]:
        if neighbor not in visited:
            result = depth_limited_search(network, neighbor, target, depth - 1, visited, path)
            if result:
                return result

    visited.remove(node)
    path.pop()
    return None

def iterative_deepening_search(network, start, target, max_depth):
    for limit in range(max_depth + 1):
        result = depth_limited_search(network, start, target, limit)
        if result:
            return result
    return None

def uniform_cost_search(network, start, target):
    pq = [(0, start, [start])]
    visited = {}

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node == target:
            return path, cost

        if node in visited and visited[node] <= cost:
            continue

        visited[node] = cost

        for neighbor, edge_cost in network.graph[node]:
            total = cost + edge_cost
            heapq.heappush(pq, (total, neighbor, path + [neighbor]))

    return None, float("inf")

# ---------------- Main Driver ---------------- #

def main():
    n, m = map(int, input("Enter number of junctions and pipes: ").split())
    network = WaterNetwork(n)

    print("\nEnter pipe connections as: node1 node2 cost")
    for _ in range(m):
        a, b, cost = map(int, input().split())
        network.add_connection(a, b, cost)

    start = int(input("\nEnter starting junction: "))
    target = int(input("Enter target junction: "))

    print("\nBreadth-First Search (ignores cost):", bfs_search(network, start, target))
    print("Depth-First Search (ignores cost):", dfs_search(network, start, target))

    limit = int(input("\nEnter depth limit for Depth-Limited Search: "))
    print("Depth-Limited Search:", depth_limited_search(network, start, target, limit))

    max_depth = int(input("\nEnter max depth for Iterative Deepening Search: "))
    print("Iterative Deepening Search:", iterative_deepening_search(network, start, target, max_depth))

    path, total = uniform_cost_search(network, start, target)
    print(f"\nUniform Cost Search: Path {path} with total cost {total}")

if __name__ == "__main__":
    main()
