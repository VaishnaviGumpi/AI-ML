# botbrain.py
# Main interface for BotBrain campus navigation agent

from campus_map import buildings, get_building_info, campus_graph
from search_algorithms import bfs, dfs, ucs, astar

def print_path(path):
    print(' -> '.join(path))
    print('Building info along the path:')
    for b in path:
        print(f"- {get_building_info(b)}")

def total_distance(path):
    dist = 0
    for i in range(len(path)-1):
        for neighbor, d, _, _ in campus_graph[path[i]]:
            if neighbor == path[i+1]:
                dist += d
                break
    return dist

def estimated_time(distance, speed=80):
    return round(distance / speed, 2)  # minutes

def normalize_building_name(name):
    name = name.strip().lower()
    for b in buildings:
        if b.lower() == name:
            return b
    return None

def main():
    print("Welcome to BotBrain - Chanakya University Campus Navigator!")
    print("Available buildings:")
    for b in buildings:
        print(f"- {b}")
    while True:
        print("\nType 'end' at any prompt to exit.")
        src_input = input("Enter source building: ")
        if src_input.strip().lower() == 'end':
            print("Exiting BotBrain. Goodbye!")
            break
        dst_input = input("Enter destination building: ")
        if dst_input.strip().lower() == 'end':
            print("Exiting BotBrain. Goodbye!")
            break
        src = normalize_building_name(src_input)
        dst = normalize_building_name(dst_input)
        if not src or not dst:
            print("Invalid source or destination building name. Please check your input.")
            continue

        print("Choose search algorithm:")
        print("1. Breadth-First Search (BFS)")
        print("2. Depth-First Search (DFS)")
        print("3. Uniform Cost Search (UCS)")
        print("4. A* Search")
        print("5. Auto (Shortest Path)")
        algo_choice = input("Enter 1/2/3/4/5: ").strip()

        if algo_choice == '1':
            path, explored = bfs(src, dst)
            dist = total_distance(path) if path else 0
            algo_used = "Breadth-First Search (BFS)"
        elif algo_choice == '2':
            path, explored = dfs(src, dst)
            dist = total_distance(path) if path else 0
            algo_used = "Depth-First Search (DFS)"
        elif algo_choice == '3':
            path, explored, dist = ucs(src, dst)
            algo_used = "Uniform Cost Search (UCS)"
        elif algo_choice == '4':
            path, explored, dist = astar(src, dst)
            algo_used = "A* Search"
        elif algo_choice == '5':
            # Use UCS and A* to find the shortest path, pick the best
            path_ucs, explored_ucs, dist_ucs = ucs(src, dst)
            path_astar, explored_astar, dist_astar = astar(src, dst)
            if path_astar and (not path_ucs or dist_astar <= dist_ucs):
                path, explored, dist, algo_used = path_astar, explored_astar, dist_astar, "A* Search (Auto)"
            elif path_ucs:
                path, explored, dist, algo_used = path_ucs, explored_ucs, dist_ucs, "Uniform Cost Search (UCS, Auto)"
            else:
                path, explored, dist, algo_used = None, [], 0, None
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
            continue

        if path:
            print(f"\n{algo_used} explored: {explored}")
            print("\nFound path:")
            print_path(path)
            print(f"Total distance: {dist} meters")
            print(f"Estimated walking time: {estimated_time(dist)} minutes")
            print(f"(Used {algo_used})")
        else:
            print("No path found.")

if __name__ == "__main__":
    main()
