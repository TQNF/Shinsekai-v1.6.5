@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0.."
echo ============================================
echo   RAG 知识库管理工具
echo   list / search / delete / clear
echo ============================================
echo.
.\.venv\Scripts\python.exe tools\rag_manager.py %*
echo.
pause
