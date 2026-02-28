@echo off

REM Build and run backend
docker build -t vn-roboadvisor-backend -f backend.Dockerfile .
docker run -d -p 8000:8000 --name robo-backend vn-roboadvisor-backend

REM Build and run frontend
docker build -t vn-roboadvisor-frontend -f frontend.Dockerfile .
docker run -d -p 3000:3000 --name robo-frontend vn-roboadvisor-frontend

echo Backend running at http://localhost:8000
echo Frontend running at http://localhost:3000
pause
