"""The Backend/API Adapter (Phase 7).

Sits strictly above `engine/` (the framework-independent scheduler
core) and below the React frontend:

    Scheduler Core (engine/*)  ->  API Adapter (this package)  ->  HTTP/WebSocket  ->  React

Nothing in `engine/` imports from this package; nothing here contains
a scheduling decision - see `api/app.py`'s module docstring.
"""
