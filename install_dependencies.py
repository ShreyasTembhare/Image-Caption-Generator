#!/usr/bin/env python3
"""
Dependency Installation Script for Advanced Image Caption Generator
This script helps resolve TensorFlow dependency conflicts.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return the result."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def install_dependencies():
    """Install dependencies with conflict resolution."""
    print("🚀 Installing dependencies for Advanced Image Caption Generator...")
    
    # Step 1: Upgrade pip
    print("\n📦 Upgrading pip...")
    success, stdout, stderr = run_command("python -m pip install --upgrade pip")
    if not success:
        print(f"⚠️ Warning: Could not upgrade pip: {stderr}")
    
    # Step 2: Uninstall conflicting TensorFlow packages
    print("\n🧹 Cleaning up conflicting TensorFlow packages...")
    packages_to_remove = [
        "tensorflow-intel",
        "tensorflow-cpu",
        "tensorflow-gpu"
    ]
    
    for package in packages_to_remove:
        success, stdout, stderr = run_command(f"pip uninstall {package} -y")
        if success:
            print(f"✅ Removed {package}")
        else:
            print(f"ℹ️ {package} not found or already removed")
    
    # Step 3: Install base requirements
    print("\n📥 Installing base requirements...")
    success, stdout, stderr = run_command("pip install streamlit pillow numpy pandas plotly")
    if success:
        print("✅ Base requirements installed successfully")
    else:
        print(f"❌ Error installing base requirements: {stderr}")
        return False
    
    # Step 4: Install TensorFlow with specific version
    print("\n🧠 Installing TensorFlow...")
    success, stdout, stderr = run_command("pip install tensorflow==2.16.1")
    if success:
        print("✅ TensorFlow installed successfully")
    else:
        print(f"⚠️ Warning: TensorFlow installation failed: {stderr}")
        print("ℹ️ The app will work with template-based generation only")
    
    # Step 5: Install additional dependencies
    print("\n📚 Installing additional dependencies...")
    additional_packages = [
        "scikit-learn>=1.3.0",
        "nltk>=3.8.0",
        "opencv-python>=4.8.0"
    ]
    
    for package in additional_packages:
        success, stdout, stderr = run_command(f"pip install {package}")
        if success:
            print(f"✅ {package} installed successfully")
        else:
            print(f"⚠️ Warning: {package} installation failed: {stderr}")
    
    print("\n🎉 Installation completed!")
    print("\n📋 Summary:")
    print("✅ Base requirements: streamlit, pillow, numpy, pandas, plotly")
    if success:
        print("✅ TensorFlow: Available for deep learning features")
    else:
        print("⚠️ TensorFlow: Not available (template-based generation only)")
    print("✅ Additional packages: scikit-learn, nltk, opencv-python")
    
    return True

def check_installation():
    """Check if all dependencies are properly installed."""
    print("\n🔍 Checking installation...")
    
    # Test imports
    imports_to_test = [
        ("streamlit", "Streamlit"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("plotly", "Plotly"),
        ("tensorflow", "TensorFlow"),
        ("sklearn", "Scikit-learn"),
        ("nltk", "NLTK"),
        ("cv2", "OpenCV")
    ]
    
    all_good = True
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {name}: Available")
        except ImportError:
            print(f"❌ {name}: Not available")
            all_good = False
    
    return all_good

def main():
    """Main installation function."""
    print("=" * 60)
    print("🤖 Advanced Image Caption Generator - Dependency Installer")
    print("=" * 60)
    
    # Install dependencies
    success = install_dependencies()
    
    # Check installation
    print("\n" + "=" * 60)
    print("🔍 Installation Verification")
    print("=" * 60)
    
    all_good = check_installation()
    
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 All dependencies installed successfully!")
        print("🚀 You can now run: streamlit run app.py")
    else:
        print("⚠️ Some dependencies may not be available.")
        print("ℹ️ The app will work with reduced functionality.")
        print("🚀 You can still run: streamlit run app.py")
    print("=" * 60)

if __name__ == "__main__":
    main() 