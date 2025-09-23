# agent_explanation.md

## BotBrain as an Intelligent Agent

### Agent Environment
BotBrain operates in a digital model of the Chanakya University campus, where buildings are nodes and paths are edges with constraints (distance, direction, speed).

### Agent Type
BotBrain is a **goal-based agent**. It takes user queries (source, destination, algorithm) and plans a path to achieve the navigation goal, rather than simply reacting to current percepts.

### PEAS Analysis
- **Performance Measure:** Path optimality (shortest/fastest), number of nodes explored, user satisfaction, correct info.
- **Environment:** Digital campus map (graph), user queries, building info.
- **Actuators:** Text output (path, info, recommendations).
- **Sensors:** User input (source, destination, algorithm), campus map data.

### Agent Behavior
BotBrain makes decisions by:
- Accepting user goals (navigation queries)
- Selecting and running a search algorithm
- Exploring the campus graph step-by-step
- Returning the best path, info, and recommendations

BotBrain can be extended to handle dynamic environments, real-time data, or visual outputs.
