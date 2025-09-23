# campus_map.py
# Campus graph representation for Chanakya University (tentative data)

import math

class Building:
    def __init__(self, name, coord, info, services=None, sublocations=None):
        self.name = name
        self.coord = coord  # (lat, lon) tuple
        self.info = info
        self.services = services or []
        self.sublocations = sublocations or []

# List of buildings (real coordinates and info)
buildings = {
    'Main Gate': Building('Main Gate', (13.221305, 77.755056), 'Campus entrance', ['Security'], sublocations=[]),
    'Admin Block': Building('Admin Block', (13.222194, 77.755250), 'Registrar office, Fee payment counter', ['Registrar', 'Fees'], sublocations=['Library', 'Auditorium', 'LG/Cafeteria']),
    'Academic Block 1': Building('Academic Block 1', (13.223333, 77.754917), 'Classrooms, Labs', ['Lecture halls']),
    'Academic Block 2': Building('Academic Block 2', (13.223389, 77.756056), 'Classrooms, Labs', ['Lecture halls']),
    'Academic Block 3': Building('Academic Block 3', (13.222389, 77.756278), 'Classrooms, Labs', ['Lecture halls']),
    'Hostel': Building('Hostel', (13.224528, 77.758167), 'Student accommodation', ['Rooms', 'Laundry'], sublocations=['Mini Mart']),
    'Food Court': Building('Food Court', (13.224861, 77.757222), 'Food court, Open 7 AM - 9 PM', ['Food', 'Snacks'], sublocations=['Gym', 'Laundry']),
    'Sports Complex': Building('Sports Complex', (13.228417, 77.758278), 'Gym, Courts', ['Gym', 'Courts'], sublocations=['Tennis Court', 'Vollyball Court', 'Basket Ball Ground', 'Cricket Ground']),
    'Central Junction': Building('Central Junction', (13.222778, 77.755611), 'Campus hub', ['Info desk']),
    'Small Junction': Building('Small Junction', (13.22485, 77.75836), 'Junction near Hostel', []),
    'Generator Room': Building('Generator Room', (13.22313, 77.75728), 'Power backup facility', []),
    'Laundry': Building('Laundry', (13.224528, 77.757056), 'Laundry services', ['Laundry']),
    'Mini Mart': Building('Mini Mart', (13.224556, 77.758250), 'Convenience store', ['Groceries']),
    # Sublocations as separate entries for info lookup, but same coordinates as parent
    'Library': Building('Library', (13.222194, 77.755250), 'Study halls available, Open 8 AM - 10 PM', ['Books', 'Study rooms']),
    'Auditorium': Building('Auditorium', (13.222194, 77.755250), 'Events, Seminars', ['Events']),
    'LG/Cafeteria': Building('LG/Cafeteria', (13.222194, 77.755250), 'Cafeteria, Snacks', ['Food']),
    'Gym': Building('Gym', (13.224861, 77.757222), 'Fitness center', ['Gym']),
    'Tennis Court': Building('Tennis Court', (13.228417, 77.758278), 'Tennis court', ['Tennis']),
    'Vollyball Court': Building('Vollyball Court', (13.228417, 77.758278), 'Volleyball court', ['Volleyball']),
    'Basket Ball Ground': Building('Basket Ball Ground', (13.228417, 77.758278), 'Basketball ground', ['Basketball']),
    'Cricket Ground': Building('Cricket Ground', (13.228417, 77.758278), 'Cricket ground', ['Cricket']),
}

# Graph: adjacency list with (neighbor, distance in meters, direction, speed)
# direction: 'two-way' or 'one-way' (from key to neighbor)
# speed: meters/minute (default 80)
campus_graph = {
    'Main Gate': [
        ('Admin Block', 120, 'two-way', 80),
        ('Generator Room', 150, 'two-way', 80),
    ],
    'Admin Block': [
        ('Main Gate', 120, 'two-way', 80),
        ('Central Junction', 60, 'two-way', 80),
        ('Library', 30, 'two-way', 80),
        ('Auditorium', 30, 'two-way', 80),
        ('LG/Cafeteria', 30, 'two-way', 80),
    ],
    'Library': [
        ('Admin Block', 30, 'two-way', 80),
    ],
    'Auditorium': [
        ('Admin Block', 30, 'two-way', 80),
    ],
    'LG/Cafeteria': [
        ('Admin Block', 30, 'two-way', 80),
    ],
    'Central Junction': [
        ('Admin Block', 60, 'two-way', 80),
        ('Academic Block 1', 100, 'two-way', 80),
        ('Academic Block 2', 100, 'two-way', 80),
        ('Academic Block 3', 100, 'two-way', 80),
        ('Small Junction', 120, 'two-way', 80),
    ],
    'Academic Block 1': [
        ('Central Junction', 100, 'two-way', 80),
    ],
    'Academic Block 2': [
        ('Central Junction', 100, 'two-way', 80),
    ],
    'Academic Block 3': [
        ('Central Junction', 100, 'two-way', 80),
    ],
    'Generator Room': [
        ('Small Junction', 50, 'two-way', 80),
        ('Hostel', 50, 'two-way', 80),
        ('Main Gate', 150, 'two-way', 80),
    ],
    'Small Junction': [
        ('Central Junction', 120, 'two-way', 80),
        ('Generator Room', 50, 'two-way', 80),
        ('Hostel', 30, 'two-way', 80),
        ('Food Court', 60, 'two-way', 80),
        ('Sports Complex', 200, 'two-way', 80),
    ],
    'Hostel': [
        ('Small Junction', 30, 'two-way', 80),
        ('Mini Mart', 30, 'two-way', 80),
        ('Generator Room', 50, 'two-way', 80),
    ],
    'Mini Mart': [
        ('Hostel', 30, 'two-way', 80),
    ],
    'Food Court': [
        ('Small Junction', 60, 'two-way', 80),
        ('Laundry', 30, 'two-way', 80),
        ('Gym', 20, 'two-way', 80),
    ],
    'Laundry': [
        ('Food Court', 30, 'two-way', 80),
    ],
    'Gym': [
        ('Food Court', 20, 'two-way', 80),
    ],
    'Sports Complex': [
        ('Small Junction', 200, 'two-way', 80),
        ('Tennis Court', 30, 'two-way', 80),
        ('Vollyball Court', 30, 'two-way', 80),
        ('Basket Ball Ground', 30, 'two-way', 80),
        ('Cricket Ground', 100, 'two-way', 80),
    ],
    'Tennis Court': [
        ('Sports Complex', 30, 'two-way', 80),
    ],
    'Vollyball Court': [
        ('Sports Complex', 30, 'two-way', 80),
    ],
    'Basket Ball Ground': [
        ('Sports Complex', 30, 'two-way', 80),
    ],
    'Cricket Ground': [
        ('Sports Complex', 100, 'two-way', 80),
    ],
}
def euclidean_distance(coord1, coord2):
    # Haversine formula for distance between two lat/lon points (in meters)
    from math import radians, sin, cos, sqrt, atan2
    R = 6371000  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2*atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_neighbors(building):
    return campus_graph.get(building, [])

def get_building_info(building):
    b = buildings.get(building)
    if b:
        info = f"{b.name}: {b.info} (Services: {', '.join(b.services)})"
        if b.sublocations:
            info += f"\n  Sublocations: {', '.join(b.sublocations)}"
        return info
    return "No info available."
# Optionally, a function to get sublocation options for a building
def get_sublocation_options(building):
    b = buildings.get(building)
    if b and b.sublocations:
        return b.sublocations
    return []
