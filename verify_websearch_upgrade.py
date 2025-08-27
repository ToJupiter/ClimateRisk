#!/usr/bin/env python3
"""
WebSearch Upgrade Verification Script

This script verifies that the DuckDuckGo search has been successfully 
replaced with OpenAI Web Search as requested.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def verify_code_changes():
    """Verify that the code has been properly updated."""
    print("🔍 Verifying Code Changes")
    print("=" * 50)
    
    # Check if OpenAI import is present in WebsiteFinder
    website_finder_path = Path("src/website_finder.py")
    if not website_finder_path.exists():
        print("❌ src/website_finder.py not found")
        return False
    
    with open(website_finder_path, 'r') as f:
        content = f.read()
    
    checks = [
        ("OpenAI import", "from openai import AsyncOpenAI" in content),
        ("DuckDuckGo removal", "duckduckgo" not in content.lower()),
        ("OpenAI client init", "AsyncOpenAI(api_key=api_key)" in content),
        ("Web search model", "gpt-4o-search-preview" in content),
        ("Geographic support", "user_location" in content),
    ]
    
    passed = 0
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}: {'PASS' if result else 'FAIL'}")
        if result:
            passed += 1
    
    print(f"\n📊 Code verification: {passed}/{len(checks)} checks passed")
    return passed == len(checks)


def verify_dependencies():
    """Verify that required dependencies are available."""
    print("\n🔧 Verifying Dependencies")
    print("=" * 30)
    
    try:
        import openai
        print(f"✅ OpenAI library: {openai.__version__}")
        
        import aiohttp
        print(f"✅ aiohttp library: {aiohttp.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


async def test_functionality():
    """Test the actual functionality."""
    print("\n🧪 Testing Functionality")
    print("=" * 25)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️ OPENAI_API_KEY not set - skipping functionality test")
        print("Set the API key to test full functionality:")
        print("export OPENAI_API_KEY='your-key-here'")
        return False
    
    try:
        from website_finder import WebsiteFinder
        
        print("✅ WebsiteFinder import successful")
        
        # Test with a simple company
        async with WebsiteFinder(api_key=api_key) as finder:
            print("✅ WebsiteFinder initialization successful")
            
            # Try a simple search
            result = await finder.find_company_website("Microsoft", "US")
            
            if result:
                print(f"✅ Web search successful: {result}")
                return True
            else:
                print("⚠️ Web search returned no results")
                return False
                
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False


def check_documentation():
    """Check if documentation has been updated."""
    print("\n📚 Checking Documentation Updates")
    print("=" * 35)
    
    docs_to_check = [
        ("README.md", "OpenAI Web Search"),
        ("IMPROVEMENTS.md", "OpenAI Web Search"),
    ]
    
    updated_docs = 0
    for doc_file, search_term in docs_to_check:
        try:
            with open(doc_file, 'r') as f:
                content = f.read()
            
            if search_term in content:
                print(f"✅ {doc_file}: Updated with {search_term}")
                updated_docs += 1
            else:
                print(f"⚠️ {doc_file}: No mention of {search_term}")
        except FileNotFoundError:
            print(f"❌ {doc_file}: File not found")
    
    return updated_docs > 0


async def main():
    """Run all verification checks."""
    print("🚀 WebSearch Upgrade Verification")
    print("=" * 60)
    print("Verifying DuckDuckGo → OpenAI Web Search migration")
    print()
    
    # Run all verification checks
    code_check = verify_code_changes()
    deps_check = verify_dependencies()
    func_check = await test_functionality()
    docs_check = check_documentation()
    
    print("\n" + "=" * 60)
    print("📋 Verification Summary")
    print("=" * 60)
    
    results = [
        ("Code Changes", code_check),
        ("Dependencies", deps_check),
        ("Functionality", func_check),
        ("Documentation", docs_check),
    ]
    
    passed = 0
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:15} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 SUCCESS: DuckDuckGo has been successfully replaced with OpenAI Web Search!")
        print("\n✅ Key Improvements:")
        print("   - More accurate website discovery")
        print("   - Up-to-date search results")
        print("   - Geographic location awareness")
        print("   - Better integration with OpenAI ecosystem")
        print("   - Cost-efficient implementation")
        
    elif passed >= 2:
        print("\n⚠️ PARTIAL SUCCESS: Most components are working")
        print("Please check any failed components above")
        
    else:
        print("\n❌ FAILED: Migration incomplete")
        print("Please review the failed checks above")
    
    print(f"\n📝 Next Steps:")
    if not func_check and not os.getenv('OPENAI_API_KEY'):
        print("1. Set OPENAI_API_KEY environment variable")
    print("2. Run: python test_openai_websearch.py")
    print("3. Run: python main.py --company 'Microsoft' --openai-key 'your-key'")


if __name__ == "__main__":
    asyncio.run(main()) 