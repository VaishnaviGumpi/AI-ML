from queue import PriorityQueue
import math

print("Checking paths in a network")
print("Applying A* search method\n")

def estimate_distance(node, goal, positions):
    """Heuristic function: straight-line distance between two points"""
    x1, y1 = positions[node]
    x2, y2 = positions[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def a_star(graph, start, goal, positions):
    """Finds a path using cost + heuristic."""
    pq = PriorityQueue()
    pq.put((0, start))

    came_from = {start: None}
    cost = {start: 0}
    visited_count = 0

    while not pq.empty():
        _, current = pq.get()
        visited_count += 1

        if current == goal:
            # backtrack path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, cost[goal], visited_count

        for neighbor, edge_cost in graph.get(current, []):
            new_cost = cost[current] + edge_cost
            if neighbor not in cost or new_cost < cost[neighbor]:
                cost[neighbor] = new_cost
                priority = new_cost + estimate_distance(neighbor, goal, positions)
                pq.put((priority, neighbor))
                came_from[neighbor] = current

    return None, float("inf"), visited_count


if "__name__" == "__main__":
    # Example graph
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 10), ('D', 2), ('E', 5)],
        'C': [('A', 4), ('F', 3)],
        'D': [('B', 2)],
        'E': [('B', 5), ('F', 1)],
        'F': [('C', 3), ('E', 1)]
    }

    positions = {
        'A': (0, 5),
        'B': (1, 3),
        'C': (4, 2),
        'D': (0, 2),
        'E': (3, 1),
        'F': (5, 0)
    }

    start, goal = 'A', 'F'
    path, total_cost, explored = a_star(graph, start, goal, positions)

    if path:
        print("Best path:", " -> ".join(path))
        print("Cost:", total_cost, "| Nodes checked:", explored)
    else:
        print("No path available")