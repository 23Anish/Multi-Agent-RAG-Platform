import uvicorn
from app.mcp.server import mcp_app
from app.config import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "app.mcp_main:mcp_app",
        host="0.0.0.0",
        port=settings.mcp_server_port,
        reload=settings.environment == "local",
    )
