import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error executing: {cmd}")
        sys.exit(result.returncode)

def main():
    print("=== Research-the-Researcher Setup ===")
    
    # 1. Install standard requirements
    print("\n[1/4] Installing standard dependencies...")
    run_cmd(f"{sys.executable} -m pip install -r requirements.txt")
    
    # 2. Install crawl4ai dependencies manually to bypass lxml downgrade issues on Python 3.14
    print("\n[2/4] Installing crawl4ai dependencies...")
    crawl_deps = [
        "aiofiles", "aiosqlite", "anyio", "unclecode-litellm", "patchright", 
        "playwright-stealth", "rank-bm25", "snowballstemmer", "pyOpenSSL", 
        "psutil", "PyYAML", "nltk", "cssselect", "fake-useragent", "chardet", 
        "brotli", "humanize", "lark", "alphashape", "shapely"
    ]
    run_cmd(f"{sys.executable} -m pip install {' '.join(crawl_deps)}")
    
    # 3. Install crawl4ai with --no-deps
    print("\n[3/4] Installing crawl4ai (bypassing strict lxml version checks)...")
    run_cmd(f"{sys.executable} -m pip install crawl4ai --no-deps")
    
    # 4. Install playwright browsers
    print("\n[4/4] Downloading Headless Chromium...")
    run_cmd(f"{sys.executable} -m playwright install chromium")
    
    print("\n=== Setup Complete! ===")
    print("You can now run: python run.py --name \"Researcher Name\"")

if __name__ == "__main__":
    main()
