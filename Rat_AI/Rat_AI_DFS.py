from collections import deque

print("Finding paths in a network")
print("Using Breadth-First Search (BFS)\n")

def bfs_path(graph, start, goal):
    """BFS to get the shortest path in an unweighted graph"""
    q = deque([[start]])
    visited = {start}

    while q:
        path = q.popleft()
        node = path[-1]

        if node == goal:
            return path, visited

        for nbr in graph.get(node, []):
            if nbr not in visited:
                visited.add(nbr)
                q.append(path + [nbr])

    return None, visited


if "__name__" == "__main__":
    # Example network
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    start, goal = 'A', 'F'
    path, visited_nodes = bfs_path(graph, start, goal)

    if path:
        print("Path found:", " -> ".join(path))
        print("Visited nodes:", visited_nodes)
    else:
        print("No path found")
