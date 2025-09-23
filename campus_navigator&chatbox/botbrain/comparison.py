# comparison.py
# Compare search algorithms on 3 source-destination pairs

from search_algorithms import bfs, dfs, ucs, astar
from botbrain import total_distance

pairs = [
    ("Main Hostel", "Library"),
    ("Academic Block A", "Canteen"),
    ("Main Gate", "Admin Block"),
]

algos = [
    ("BFS", bfs),
    ("DFS", dfs),
    ("UCS", ucs),
    ("A*", astar),
]

def compare():
    print("Algorithm Comparison Table:")
    print("{:<20} {:<20} {:<8} {:<15} {:<10}".format('Source', 'Destination', 'Algo', 'Nodes Explored', 'Distance'))
    for src, dst in pairs:
        for name, algo in algos:
            if name in ["UCS", "A*"]:
                path, explored, dist = algo(src, dst)
            else:
                path, explored = algo(src, dst)
                dist = total_distance(path) if path else 0
            print(f"{src:<20} {dst:<20} {name:<8} {len(explored):<15} {dist:<10}")
    print("\nShortest path is always found by UCS and A* (if heuristic is admissible). BFS may also find shortest in unweighted graphs.")

if __name__ == "__main__":
    compare()
