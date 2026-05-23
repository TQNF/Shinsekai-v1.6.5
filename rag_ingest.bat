@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo ============================================
echo   小说原文 RAG 入库工具
echo   将小说分段、向量化后存入 Qdrant 数据库
echo   重复运行会自动清空旧数据再重新入库
echo ============================================
echo.
.\.venv\Scripts\python.exe tools\rag_ingest.py %*
echo.
pause
