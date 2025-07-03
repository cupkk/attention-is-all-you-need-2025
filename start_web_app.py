#!/usr/bin/env python3
"""
德语翻译Web应用启动脚本
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """检查必要的依赖"""
    required_files = [
        "transformer_model.pth",
        "multi30k_processed_bpe/vocab_de.pth", 
        "multi30k_processed_bpe/vocab_en.pth",
        "bpe_models/bpe_de.model",
        "bpe_models/bpe_en.model"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ 缺少必要的模型文件:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n请确保已完成模型训练并生成了所有必要文件。")
        print("参考 README.md 中的训练步骤。")
        return False
    
    print("✅ 所有模型文件检查通过")
    return True

def install_dependencies():
    """安装Python依赖"""
    print("📦 检查并安装Python依赖...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-webapp.txt"], 
                      check=True, capture_output=True)
        print("✅ 依赖安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def start_app():
    """启动Web应用"""
    print("🚀 启动德语翻译Web应用...")
    print("=" * 50)
    print("🌐 应用将在以下地址启动:")
    print("   本地访问: http://localhost:5000")
    print("   局域网访问: http://0.0.0.0:5000")
    print("=" * 50)
    print("💡 使用提示:")
    print("   - 在浏览器中打开上述地址")
    print("   - 输入德语句子进行翻译")
    print("   - 按 Ctrl+C 停止服务")
    print("=" * 50)
    
    os.chdir("flask_app")
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 应用启动失败: {e}")

def main():
    print("🇩🇪 德语翻译Web应用启动器")
    print("=" * 40)
    
    # 检查模型文件
    if not check_dependencies():
        return
    
    # 安装依赖
    if not install_dependencies():
        return
    
    # 启动应用
    start_app()

if __name__ == "__main__":
    main()
