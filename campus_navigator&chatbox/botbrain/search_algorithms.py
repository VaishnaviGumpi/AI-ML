# search_algorithms.py
# Implements BFS, DFS, UCS, and A* for campus navigation

from collections import deque
import heapq
from campus_map import buildings, campus_graph, euclidean_distance, get_neighbors

def bfs(start, goal):
    queue = deque([[start]])
    visited = set()
    explored = []
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal:
            return path, explored
        if node not in visited:
            visited.add(node)
            explored.append(node)
            for neighbor, _, direction, _ in get_neighbors(node):
                if direction == 'one-way' and neighbor not in campus_graph.get(node, []):
                    continue
                if neighbor not in visited:
                    queue.append(path + [neighbor])
    return None, explored

def dfs(start, goal):
    stack = [[start]]
    visited = set()
    explored = []
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == goal:
            return path, explored
        if node not in visited:
            visited.add(node)
            explored.append(node)
            for neighbor, _, direction, _ in reversed(get_neighbors(node)):
                if direction == 'one-way' and neighbor not in campus_graph.get(node, []):
                    continue
                if neighbor not in visited:
                    stack.append(path + [neighbor])
    return None, explored

def ucs(start, goal):
    heap = [(0, [start])]
    visited = set()
    explored = []
    while heap:
        cost, path = heapq.heappop(heap)
        node = path[-1]
        if node == goal:
            return path, explored, cost
        if node not in visited:
            visited.add(node)
            explored.append(node)
            for neighbor, dist, direction, _ in get_neighbors(node):
                if direction == 'one-way' and neighbor not in campus_graph.get(node, []):
                    continue
                if neighbor not in visited:
                    heapq.heappush(heap, (cost + dist, path + [neighbor]))
    return None, explored, float('inf')

def astar(start, goal):
    heap = [(euclidean_distance(buildings[start].coord, buildings[goal].coord), 0, [start])]
    visited = set()
    explored = []
    while heap:
        est_total, cost, path = heapq.heappop(heap)
        node = path[-1]
        if node == goal:
            return path, explored, cost
        if node not in visited:
            visited.add(node)
            explored.append(node)
            for neighbor, dist, direction, _ in get_neighbors(node):
                if direction == 'one-way' and neighbor not in campus_graph.get(node, []):
                    continue
                if neighbor not in visited:
                    g = cost + dist
                    h = euclidean_distance(buildings[neighbor].coord, buildings[goal].coord)
                    heapq.heappush(heap, (g + h, g, path + [neighbor]))
    return None, explored, float('inf')
