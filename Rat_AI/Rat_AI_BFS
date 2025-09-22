from collections import deque

print("Checking routes in a network")
print("Applying BFS (Breadth-First Search)\n")

def bfs_search(graph, start, goal):
    """Finds a path using BFS approach"""
    q = deque([[start]])
    seen = {start}

    while q:
        path = q.popleft()
        current = path[-1]

        if current == goal:
            return path, seen

        for neighbor in graph.get(current, []):
            if neighbor not in seen:
                seen.add(neighbor)
                q.append(path + [neighbor])

    return None, seen


if "__name__" == "__main__":
    # Example graph
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    start, goal = 'A', 'F'
    path, visited_nodes = bfs_search(graph, start, goal)

    if path:
        print("Path found:", " -> ".join(path))
        print("Visited:", visited_nodes)
    else:
        print("No path exists")
