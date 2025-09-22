import heapq
import math
import time


class Point:
    __slots__ = ['parent', 'pos', 'g', 'h', 'f']
    def __init__(self, position, parent=None):
        self.parent = parent
        self.pos = position
        self.g = 0 # Cost from start
        self.h = 0 # Heuristic cost
        self.f = 0 # Total cost

    def __lt__(self, other):
        return self.f < other.f
    
    def __eq__(self, other):
        return self.pos == other.pos

def build_path(node):

    path = []
    while node:
        path.append(node.pos)
        node = node.parent
    return path[::-1]

def find_path_greedy(maze, start_pos, goal_pos):

    start_node = Point(start_pos)
    goal_node = Point(goal_pos)

    pq = [] 
    heapq.heappush(pq, start_node)
    
    visited = set() 
    nodes_explored = 0

    while pq:
        curr = heapq.heappop(pq)
        nodes_explored += 1
        
  

        if curr == goal_node:
            return build_path(curr), len(build_path(curr)) - 1, nodes_explored

        if curr.pos in visited:
            continue
        visited.add(curr.pos)

  
        for move in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            next_pos = (curr.pos[0] + move[0], curr.pos[1] + move[1])

            if not (0 <= next_pos[0] < len(maze) and 0 <= next_pos[1] < len(maze[0])) or maze[next_pos[0]][next_pos[1]] == '1':
                continue
            
            neighbor = Point(next_pos, parent=curr)
            
            dx = abs(neighbor.pos[0] - goal_node.pos[0])
            dy = abs(neighbor.pos[1] - goal_node.pos[1])
            neighbor.f = dx + dy 
            heapq.heappush(pq, neighbor)

    return None, 0, nodes_explored

def find_path_astar(maze, start_pos, goal_pos, ghost_zones):
    """Finds a path using A* Search. Slower but optimal."""
    start_node = Point(start_pos)
    goal_node = Point(goal_pos)

    pq = []
    heapq.heappush(pq, start_node)
    

    visited_costs = {start_pos: 0} 
    nodes_explored = 0

    while pq:
        curr = heapq.heappop(pq)
        nodes_explored += 1

        if curr == goal_node:
            return build_path(curr), len(build_path(curr)) - 1, nodes_explored

        
        for move in [(0,1,1), (0,-1,1), (1,0,1), (-1,0,1), (1,1,math.sqrt(2)), (1,-1,math.sqrt(2)), (-1,1,math.sqrt(2)), (-1,-1,math.sqrt(2))]:
            next_pos = (curr.pos[0] + move[0], curr.pos[1] + move[1])
            move_cost = move[2]

            if not (0 <= next_pos[0] < len(maze) and 0 <= next_pos[1] < len(maze[0])) or maze[next_pos[0]][next_pos[1]] == '1':
                continue

            neighbor = Point(next_pos, parent=curr)

            
            g_cost = curr.g + move_cost
            if neighbor.pos in ghost_zones:
                g_cost += ghost_zones[neighbor.pos]
            
      
            if neighbor.pos in visited_costs and visited_costs[neighbor.pos] <= g_cost:
                continue
            
            visited_costs[neighbor.pos] = g_cost
            
           
            dx = abs(neighbor.pos[0] - goal_node.pos[0])
            dy = abs(neighbor.pos[1] - goal_node.pos[1])
            h_cost = dx + dy
            
            neighbor.g = g_cost
            neighbor.f = g_cost + h_cost

            heapq.heappush(pq, neighbor)
            
    return None, 0, nodes_explored


if __name__ == "__main__":
    
    the_map = [
        ['S', '0', '0', '1', '0', '0'],
        ['1', '1', '0', '1', 'G', '0'],
        ['0', '0', '0', '1', '0', '0'],
        ['0', '1', '1', 'Z', '1', '1'],
        ['0', '0', '0', 'Z', '0', '0']
    ]
    start = (0, 0)
    goal = (1, 4)
    ghost_penalties = {(3, 3): 10, (4, 3): 10}

    
    greedy_path, greedy_len, greedy_exp = find_path_greedy(the_map, start, goal)
    
    
    astar_path, astar_len, astar_exp = find_path_astar(the_map, start, goal, ghost_penalties)
    
    
    print("--- Pathfinding Results ---")
    print("\nGreedy Best-First Search")
    if greedy_path:
        print(f"  Path Length:   {greedy_len} steps")
        print(f"  Nodes Checked: {greedy_exp}")
        print(f"  Path Found:    {greedy_path}")
    
    print("\nA* Search")
    if astar_path:
        print(f"  Path Length:   {astar_len} steps")
        print(f"  Nodes Checked: {astar_exp}")
        print(f"  Path Found:    {astar_path}")
    
