"""
System Test Script

Quick test to verify all components are working correctly.
"""

import sys
import os
import importlib.util
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.insert(0, src_path)
    
    modules_to_test = [
        'website_finder',
        'enhanced_crawler', 
        'report_finder',
        'openai_selector',
        'report_downloader'
    ]
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name}: OK")
        except ImportError as e:
            print(f"❌ {module_name}: FAILED - {e}")
            return False
    
    return True


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'main.py',
        'config.json',
        'requirements.txt',
        'README.md',
        'src/website_finder.py',
        'src/enhanced_crawler.py',
        'src/report_finder.py', 
        'src/openai_selector.py',
        'src/report_downloader.py',
        'data/2020_ESG-SCORES-28-COUNTRIES.csv'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}: EXISTS")
        else:
            print(f"❌ {file_path}: MISSING")
            all_exist = False
    
    return all_exist


def test_config():
    """Test that configuration is valid."""
    print("\nTesting configuration...")
    
    try:
        import json
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        required_keys = ['target_years', 'crawler', 'download']
        for key in required_keys:
            if key in config:
                print(f"✅ config.{key}: OK")
            else:
                print(f"❌ config.{key}: MISSING")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ config.json: INVALID - {e}")
        return False


def test_output_directory():
    """Test that output directory structure can be created."""
    print("\nTesting output directory...")
    
    try:
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        # Test subdirectories
        (output_dir / "reports" / "2023").mkdir(parents=True, exist_ok=True)
        (output_dir / "metadata").mkdir(exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        
        print("✅ Output directory structure: OK")
        
        # Cleanup
        import shutil
        shutil.rmtree(output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Output directory: FAILED - {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 System Test - Annual Report Crawler")
    print("=" * 50)
    
    tests = [
        test_file_structure,
        test_imports,
        test_config,
        test_output_directory
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run demo: python demo.py")
        print("3. Run crawler: python main.py --help")
    else:
        print("❌ Some tests failed. Please check the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
