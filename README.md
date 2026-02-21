# 🇻🇳 VN Robo-Advisor

A quantitative portfolio tracker powered by Markowitz Mean-Variance Optimization, the Black-Litterman model, and Monte Carlo risk simulations, specifically built for the Vietnamese stock market.

## Architecture

This project was recently migrated from a monolithic Streamlit application to a modern, fully decoupled web architecture:

- **Backend (FastAPI)**: Modular Python quantitative core. Exposes REST endpoints for portfolio operations, market data fetching, algorithmic rebalancing, and complex analytics (Efficient Frontier, Rolling Sharpe).
- **Frontend (Next.js)**: Modern React application providing a responsive, dark-mode, glassmorphism UI. It uses React Query for automatic background data refresh and Recharts for interactive analytics visualizations. Data is fetched from the FastAPI backend.

## How to Run

The easiest way to run both the frontend and backend simultaneously is using Docker Compose.

### Option 1: Docker Compose (Recommended)

Make sure you have [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running on your system.

From the root directory of the project, run:
```bash
docker compose up -d --build
```

This commands builds both images and starts both containers in the background.
- Frontend UI: http://localhost:3000
- Backend API Docs: http://localhost:8000/docs

To stop the services, run:
```bash
docker compose down
```

### Option 2: Local Development (Manual Start)

If you prefer to run the services natively for development:

**1. Start the FastAPI Backend**:
Open a terminal in the project root:
```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

**2. Start the Next.js Frontend**:
Open a second terminal:
```bash
cd frontend
npm install
npm run dev
```
Navigate to http://localhost:3000 in your browser.

## Features Built
- 📊 **Holdings Dashboard**: Real-time NAV tracking and sector allocation breakdown.
- 📈 **Performance Tracker**: Equity curve vs VN30 ETF benchmark, rolling 60-day Sharpe ratio, and historical market stress tests.
- 🤖 **Robo-Advisor Terminal**: Sharpe-maximizing Markowitz optimizer, Black-Litterman expected return overrides, and Efficient Frontier scatter visualizations.
