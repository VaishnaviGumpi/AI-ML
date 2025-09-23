

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'botbrain')))
from campus_map import buildings, get_building_info, campus_graph
from botbrain import normalize_building_name, total_distance, estimated_time
from search_algorithms import ucs, astar, bfs, dfs
from dotenv import load_dotenv
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyCiDYUXvOCb5_IKhm1n5DpJaI3ZZy-1j5A"
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class NavRequest(BaseModel):
    start: str
    end: str
    algorithm: str

def extract_navigation_request(text):
    import re
    nav_patterns = [
        r"(?:navigate|go|route|path|help me (?:go|get|navigate)?) (?:to|to|get to|go to|navigate to|route to|path to)? ([\w\s]+) from ([\w\s]+)",
        r"from ([\w\s]+) to ([\w\s]+)",
        r"to ([\w\s]+) from ([\w\s]+)",
        r"([\w\s]+) from ([\w\s]+)"
    ]
    for pat in nav_patterns:
        m = re.search(pat, text.lower())
        if m:
            if pat.startswith("from"):
                src, dst = m.group(1).strip(), m.group(2).strip()
            elif pat.startswith("to"):
                dst, src = m.group(1).strip(), m.group(2).strip()
            else:
                dst, src = m.group(1).strip(), m.group(2).strip()
            return src, dst
    return None, None

def get_path_coords(path):
    coords = []
    for b in path:
        building = buildings.get(b)
        if building:
            coords.append([building.coord[0], building.coord[1]])
    return coords


# Navigation endpoint with algorithm selection
@app.post("/navigate")
def navigate_endpoint(req: NavRequest):
    start = normalize_building_name(req.start)
    end = normalize_building_name(req.end)
    algo = req.algorithm.strip().lower()
    if not start or not end:
        return {"response": "Invalid source or destination.", "path_found": False}
    if algo == "a*":
        path, explored, dist = astar(start, end)
    elif algo == "ucs":
        path, explored, dist = ucs(start, end)
    elif algo == "bfs":
        path, explored = bfs(start, end)
        dist = total_distance(path) if path else 0
    elif algo == "dfs":
        path, explored = dfs(start, end)
        dist = total_distance(path) if path else 0
    else:
        # Default to A*
        path, explored, dist = astar(start, end)
    if path:
        coords = get_path_coords(path)
        details = f"<span style='background:#e0e7ff;padding:4px 8px;border-radius:6px;font-weight:600;'>Shortest path from {start} to {end}:</span><br>"
        details += " → ".join(path)
        details += f"<br><span style='background:#e0e7ff;padding:4px 8px;border-radius:6px;'>Distance: {dist} meters | Estimated time: {estimated_time(dist)} min</span>"
        details += "<br><b>Details:</b><br>"
        for b in path:
            details += get_building_info(b) + "<br>"
        return {
            "response": f"Here is the best route from {start} to {end}. Distance: {dist} meters. Estimated walking time: {estimated_time(dist)} min.",
            "path_found": True,
            "coords": coords,
            "details": details
        }
    else:
        return {"response": "No path found.", "path_found": False}

# Gemini chat endpoint using Gemini Flash 2.0
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    user_msg = req.message
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(user_msg)
        answer = response.text if hasattr(response, "text") else str(response)
        return {"response": answer, "path_found": False}
    except Exception as e:
        return {"response": f"Gemini API error: {str(e)}", "path_found": False}

@app.get("/")
def root():
    return {"message": "Backend is running!"}
