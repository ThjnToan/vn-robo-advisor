from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import portfolio, analytics, market

app = FastAPI(
    title="VN Robo-Advisor API",
    description="Quantitative backend for the VN Robo-Advisor portfolio management tool.",
    version="1.0.0"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://vn-robo-advisor.vercel.app"
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(market.router, prefix="/api/market", tags=["Market"])

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "vn-robo-advisor-api"}

# To run the server:
# cd /path/to/VN_RoboAdvisor
# uvicorn backend.main:app --reload
