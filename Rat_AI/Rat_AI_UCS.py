from queue import PriorityQueue

print("Finding paths in a network")
print("Using Uniform Cost Search (UCS)\n")

def ucs_search(graph, start, goal):
    """UCS to find the least-cost path"""
    pq = PriorityQueue()
    pq.put((0, start))

    came_from = {start: None}
    cost = {start: 0}
    visited = set()

    while not pq.empty():
        curr_cost, node = pq.get()
        visited.add(node)

        if node == goal:
            # backtrack to build path
            path = []
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, cost[goal], visited

        for nbr, edge_cost in graph.get(node, []):
            new_cost = cost[node] + edge_cost
            if nbr not in cost or new_cost < cost[nbr]:
                cost[nbr] = new_cost
                pq.put((new_cost, nbr))
                came_from[nbr] = node

    return None, float("inf"), visited


if _name_ == "_main_":
    # Example graph
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 10), ('D', 2), ('E', 5)],
        'C': [('A', 4), ('F', 3)],
        'D': [('B', 2)],
        'E': [('B', 5), ('F', 1)],
        'F': [('C', 3), ('E', 1)]
    }

    start, goal = 'A', 'F'
    path, total_cost, visited_nodes = ucs_search(graph, start, goal)

    if path:
        print("Best path:", " -> ".join(path))
        print("Total cost:", total_cost)
        print("Visited:", visited_nodes)
    else:
        print("No path found")
