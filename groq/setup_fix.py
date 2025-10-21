"""
Setup script to fix all dependencies and prepare the iPsychiatrist app.
Run this before starting the Streamlit app.
"""
import subprocess
import sys
import os
import shutil

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"[*] {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"[OK] Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error: {description}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    print("\n" + "="*60)
    print("iPsychiatrist Setup and Fix Script")
    print("="*60)
    
    # Step 1: Remove old vectors folder
    vectors_path = "vectors"
    if os.path.exists(vectors_path):
        print(f"\n[*] Removing old vectors folder: {vectors_path}")
        try:
            shutil.rmtree(vectors_path)
            print("[OK] Old vectors removed successfully")
        except Exception as e:
            print(f"[WARNING] Could not remove vectors folder: {e}")
    
    # Step 2: Install/upgrade all required packages
    packages = [
        "langchain>=1.0.0",
        "langchain-core>=1.0.0",
        "langchain-community>=0.4",
        "langchain-openai>=1.0.0",
        "langchain-text-splitters>=1.0.0",
        "langchain-groq>=1.0.0",
        "langchain-classic>=1.0.0",
        "langchain-huggingface>=1.0.0",
        "sentence-transformers>=5.0.0",
        "torch",
        "transformers",
        '"numpy<2,>=1.19.3"',
        '"pillow<11,>=7.1.0"',
        '"packaging<25,>=16.8"',
        "scikit-learn>=1.7.0",
        "faiss-cpu",
        "pypdf",
        "python-dotenv",
        "streamlit",
        "groq",
    ]
    
    print("\n[*] Installing/upgrading required packages...")
    packages_str = " ".join(packages)
    
    # Use py -3.12 to ensure we're using Python 3.12
    success = run_command(
        f'py -3.12 -m pip install --upgrade {packages_str}',
        "Installing all required packages"
    )
    
    if not success:
        print("\n[WARNING] Some packages may have failed to install.")
        print("This might be because Streamlit is currently running.")
        print("Please close all Streamlit instances and run this script again.")
        return False
    
    # Step 3: Verify installations
    print("\n[*] Verifying installations...")
    verification_code = """
import sys
try:
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import PyPDFLoader
    import streamlit
    print("[OK] All imports successful!")
    sys.exit(0)
except Exception as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)
"""
    
    result = subprocess.run(
        ['py', '-3.12', '-c', verification_code],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        print("\n[WARNING] Some imports failed. Please check the error messages above.")
        return False
    
    # Step 4: Replace old app.py with fixed version
    print("\n[*] Backing up and replacing app.py...")
    if os.path.exists("app.py"):
        try:
            shutil.copy("app.py", "app_backup.py")
            print("[OK] Backed up app.py to app_backup.py")
        except Exception as e:
            print(f"[WARNING] Could not backup app.py: {e}")
    
    if os.path.exists("app_fixed.py"):
        try:
            shutil.copy("app_fixed.py", "app.py")
            print("[OK] Replaced app.py with fixed version")
        except Exception as e:
            print(f"[ERROR] Could not replace app.py: {e}")
            return False
    
    print("\n" + "="*60)
    print("Setup completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Make sure your .env file has GROQ_API_KEY set")
    print("2. Run: streamlit run app.py")
    print("3. The first run will download the embedding model and create vectors")
    print("\nIf you encounter any issues, check that:")
    print("   - No other Streamlit instances are running")
    print("   - Your PDF file exists at the specified path")
    print("   - Your GROQ_API_KEY is valid")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
