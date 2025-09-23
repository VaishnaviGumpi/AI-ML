import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from college_locations import LOCATIONS

st.set_page_config(page_title="Campus Mate - Your Campus Navigator", layout="wide")

# Custom CSS for dark theme and highlighted headings
st.markdown("""
    <style>
    /* General styles */
    div.stApp, div[data-testid="stAppViewContainer"], div[data-testid="stHeader"] { 
        background-color: #121212 !important;
        color: #e0e0e0 !important;
    }
    div[data-testid="stToolbar"] {
        background-color: #1e1e1e !important;
    }
    .main .block-container {
        background-color: #121212 !important;
    }
    
    /* Container styling */
    .highlight-sub {
        background: #333;
        color: #e0e0e0;
        padding: 0.6em 1em;
        border-radius: 8px;
        font-size: 1.2em;
        font-weight: 600;
        margin-bottom: 1em;
        text-align: left;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #6366f1;
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        border: none;
        font-weight: 600;
        transition: background-color 0.2s;
        width: 100%;
        margin-top: 10px;
    }
    .stButton>button:hover {
        background-color: #4f46e5;
    }

    /* Chat styling */
    .floating-chat-toggle {
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 1000;
    }
    .floating-chat-toggle .stButton button {
        background-color: #6366f1;
        color: white;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 1.5em;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .floating-chat-toggle .stButton button:hover {
        background-color: #4f46e5;
    }

    .chat-popup {
        position: fixed;
        left: 20px;
        bottom: 80px;
        z-index: 1000;
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        width: 380px;
        max-height: 500px;
        overflow-y: auto;
        padding: 1.2rem;
        border: 1px solid #333;
    }

    .chat-messages {
        background-color: #2a2a2a;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        overflow-y: auto;
    }
    
    /* Main heading */
    .highlight-heading {
        background: linear-gradient(90deg, #6366f1 0%, #818cf8 100%);
        color: #fff;
        padding: 0.8em 1em;
        border-radius: 10px;
        font-size: 1.8em;
        font-weight: 700;
        margin-bottom: 2em;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }

    /* Form styling */
    .stSelectbox>div>div {
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
        border-radius: 8px;
        border: 1px solid #444;
    }
    .stSelectbox>div>div:hover {
        border-color: #6366f1;
    }
    .stSelectbox label {
        color: #e0e0e0;
    }

    .stTextInput>div>div>input {
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
        border-radius: 8px;
        border: 1px solid #444;
        padding: 0.7em 1em;
    }
    .stTextInput>div>div>input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 0.1rem #6366f1;
    }
    
    /* Fix highlight spans */
    .stMarkdown span[style*="background: rgb(224, 231, 255)"] {
        color: #1e1e1e !important;
        background: #e0e7ff !important;
    }
    
    /* Clean up empty containers */
    .element-container:empty,
    div[data-testid="stMarkdownContainer"] > div:empty {
        display: none;
    }
    
    /* Error messages */
    .stAlert {
        border-radius: 8px;
        background-color: #4a1c1c;
        color: #ffcccc;
        border: 1px solid #7a2828;
    }

    /* Chat messages */
    .stMarkdown p {
        margin-bottom: 0.5rem;
    }
    .stMarkdown b {
        color: #818cf8;
    }
    
    /* Chat controls */
    .chat-controls {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .chat-controls .stButton > button {
        margin-top: 0;
        background-color: #4a1c1c;
    }
    .chat-controls .stButton > button:hover {
        background-color: #7a2828;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [
        ("Hi there!", "Hi! How can I help you today?"),
        ("Hi", "I'm doing well, thanks for asking! How are you doing today?")
    ]
if 'path_data' not in st.session_state:
    st.session_state['path_data'] = None
if 'show_chat' not in st.session_state:
    st.session_state['show_chat'] = False
if 'chat_input_key' not in st.session_state:
    st.session_state['chat_input_key'] = 0

def send_message():
    current_chat_input_value = st.session_state.get(f"chat_input_{st.session_state['chat_input_key']}")
    if current_chat_input_value:
        message = current_chat_input_value
        try:
            resp = requests.post("http://localhost:8000/chat", json={"message": message})
            resp.raise_for_status()
            data = resp.json()
            answer = data.get("response", "No response from backend.")
        except requests.exceptions.ConnectionError:
            answer = "Chatbot backend is not running. Please ensure your backend server is active."
        except requests.exceptions.RequestException as e:
            answer = f"An error occurred with the chatbot backend: {e}"
        
        st.session_state['chat_history'].append((message, answer))
        st.session_state['chat_input_key'] += 1
        st.rerun()

# Main layout
st.markdown('<div class="highlight-heading">Campus Mate - Your Campus Navigator</div>', unsafe_allow_html=True)

# Navigation and Map columns
col1, col2 = st.columns([1, 2])

with col1:
    with st.container():
        st.markdown('<div class="highlight-sub">Find Your Path</div>', unsafe_allow_html=True)
        
        loc_names = [loc["name"] for loc in LOCATIONS]
        start_loc = st.selectbox("Start Location:", loc_names, key="start_loc")
        end_loc = st.selectbox("End Location:", loc_names, key="end_loc")
        algo = st.selectbox("Algorithm:", ["Select Algorithm", "A*", "UCS", "BFS", "DFS"], key="algo")
        
        if st.button("Find Path", key="find_path", help="Find the best route"):
            if start_loc == end_loc:
                st.session_state['path_data'] = None
                st.error("Start and End locations cannot be the same.")
            elif algo == "Select Algorithm":
                st.session_state['path_data'] = None
                st.error("Please select an algorithm.")
            else:
                try:
                    payload = {"start": start_loc, "end": end_loc, "algorithm": algo}
                    resp = requests.post("http://localhost:8000/navigate", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("path_found"):
                        st.session_state['path_data'] = {
                            "coords": data["coords"],
                            "details": data["details"]
                        }
                    else:
                        st.session_state['path_data'] = None
                        st.error(data.get("details", "No path found."))
                except requests.exceptions.ConnectionError:
                    st.session_state['path_data'] = None
                    st.error("Navigation backend is not running. Please ensure your backend server is active.")
                except requests.exceptions.RequestException as e:
                    st.session_state['path_data'] = None
                    st.error(f"An error occurred with the navigation backend: {e}")
                st.rerun()

        if st.session_state['path_data']:
            st.markdown('<div class="highlight-sub">Navigation Details</div>', unsafe_allow_html=True)
            st.markdown(st.session_state['path_data']['details'], unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="highlight-sub">Campus Map</div>', unsafe_allow_html=True)
        
        campus_center = [13.223333, 77.755611]
        m = folium.Map(location=campus_center, zoom_start=17, control_scale=True)
        
        for loc in LOCATIONS:
            folium.Marker(
                location=[loc["lat"], loc["lng"]],
                popup=loc["name"],
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)
        
        if st.session_state['path_data'] and st.session_state['path_data']['coords']:
            path_coords = st.session_state['path_data']['coords']
            if path_coords:
                folium.PolyLine(path_coords, color="red", weight=6, opacity=0.8).add_to(m)
                folium.Marker(path_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
                folium.Marker(path_coords[-1], popup="Destination", icon=folium.Icon(color="red")).add_to(m)
                m.fit_bounds([[min(p[0] for p in path_coords), min(p[1] for p in path_coords)], 
                              [max(p[0] for p in path_coords), max(p[1] for p in path_coords)]])

        st_folium(m, width="100%", height=500, key="campus_map")

# Chat interface
st.markdown('<div class="floating-chat-toggle">', unsafe_allow_html=True)
if st.button("💬", key="toggle_chat_btn", help="Open/Close chat"):
    st.session_state['show_chat'] = not st.session_state.get('show_chat', False)
st.markdown('</div>', unsafe_allow_html=True)

if st.session_state['show_chat']:
    with st.container():
        st.markdown('<div class="chat-popup">', unsafe_allow_html=True)
        st.markdown('<div class="highlight-sub">Campus Mate Chatbot</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
        for user_msg, bot_msg in st.session_state['chat_history']:
            st.markdown(f"**You:** {user_msg}")
            st.markdown(f"**Campus Mate:** {bot_msg}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.text_input("Campus Mate:", key=f"chat_input_{st.session_state['chat_input_key']}", 
                      on_change=send_message, placeholder="Ask me anything!", label_visibility="collapsed")
        
        st.markdown('</div>', unsafe_allow_html=True)