"""RunForm Backend Server Launcher.

Launches the FastAPI application serving the React frontend and video analysis API.
"""

from __future__ import annotations

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "runform.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
