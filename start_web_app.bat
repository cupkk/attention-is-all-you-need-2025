@echo off
chcp 65001 >nul
title 德语翻译Web应用

echo.
echo 🇩🇪 德语翻译Web应用启动器
echo ========================================

echo 📦 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 未找到Python环境
    echo 请确保已安装Python 3.7+并添加到PATH
    pause
    exit /b 1
)

echo ✅ Python环境检查通过

echo.
echo 🚀 启动Web应用...
python start_web_app.py

pause
