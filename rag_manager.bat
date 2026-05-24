@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
echo ============================================
echo   RAG + Memory Manager
echo ============================================
echo.
if "%~1"=="" (
    echo Commands:
    echo   list              List novel chunks
    echo   list --conv       List conversation history
    echo   list --mem [NAME] List mem0 memories
    echo   search WORD       Search novel
    echo   search WORD --conv   Search conversations
    echo   search WORD --mem    Search mem0
    echo   delete ID         Delete by ID
    echo   delete ID --mem   Delete mem0 by ID
    echo   clear             Clear novel database
    echo   clear --conv      Clear conversation database
    echo   clear --mem [NAME] Clear mem0 memories
    echo   purge WORD        Delete from ALL stores
    echo.
    set /p CMD=Enter command: 
    .\.venv\Scripts\python.exe tools\rag_manager.py %CMD%
) else (
    .\.venv\Scripts\python.exe tools\rag_manager.py %*
)
echo.
pause
