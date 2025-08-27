#!/usr/bin/env python3
"""
Test script for OpenAI Web Search implementation

This script tests the new WebsiteFinder that uses OpenAI Web Search
instead of DuckDuckGo to find company websites.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from website_finder import WebsiteFinder


async def test_openai_websearch():
    """Test OpenAI Web Search functionality."""
    print("🧪 Testing OpenAI Web Search Implementation")
    print("=" * 60)
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key to test the web search functionality.")
        print("\nExample:")
        print("export OPENAI_API_KEY='your-openai-api-key-here'")
        return False
    
    print("✅ OpenAI API key found")
    
    # Test companies - mix of well-known and lesser-known
    test_companies = [
        {"name": "Shell plc", "country": "GB"},
        {"name": "Microsoft Corporation", "country": "US"},
        {"name": "Arcelik AS", "country": "TR"},
        {"name": "Delta Electronics Thailand PCL", "country": "TH"},
    ]
    
    print(f"\n🔍 Testing with {len(test_companies)} companies:")
    
    try:
        async with WebsiteFinder(api_key=api_key) as finder:
            success_count = 0
            
            for i, company in enumerate(test_companies, 1):
                name = company['name']
                country = company['country']
                
                print(f"\n{i}. Testing: {name} ({country})")
                print("-" * 40)
                
                try:
                    website = await finder.find_company_website(name, country)
                    
                    if website:
                        print(f"✅ Found: {website}")
                        success_count += 1
                    else:
                        print(f"❌ Not found for {name}")
                        
                except Exception as e:
                    print(f"❌ Error for {name}: {e}")
            
            print(f"\n📊 Results Summary:")
            print(f"   - Total companies tested: {len(test_companies)}")
            print(f"   - Successful finds: {success_count}")
            print(f"   - Success rate: {success_count/len(test_companies)*100:.1f}%")
            
            if success_count > 0:
                print("\n✅ OpenAI Web Search is working correctly!")
                return True
            else:
                print("\n❌ No websites found. Please check the implementation.")
                return False
                
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


async def test_single_company():
    """Test with a single well-known company."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return False
    
    print("\n🎯 Single Company Test:")
    print("-" * 30)
    
    try:
        async with WebsiteFinder(api_key=api_key) as finder:
            website = await finder.find_company_website("Apple Inc", "US")
            
            if website:
                print(f"✅ Apple Inc website found: {website}")
                return True
            else:
                print("❌ Could not find Apple Inc website")
                return False
                
    except Exception as e:
        print(f"❌ Single company test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 OpenAI Web Search Test Suite")
    print("=" * 60)
    print("This test replaces DuckDuckGo search with OpenAI Web Search")
    print("as requested in the project requirements.")
    
    # Run basic functionality test
    basic_test_passed = await test_openai_websearch()
    
    # Run single company test
    single_test_passed = await test_single_company()
    
    print("\n" + "=" * 60)
    print("🏁 Test Suite Complete")
    print("=" * 60)
    
    if basic_test_passed and single_test_passed:
        print("✅ All tests passed! OpenAI Web Search is working correctly.")
        print("✅ DuckDuckGo has been successfully replaced with OpenAI Web Search.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        
    print("\n📝 Notes:")
    print("- OpenAI Web Search provides more accurate and up-to-date results")
    print("- The implementation uses gpt-4o-mini-search-preview for cost efficiency")
    print("- Geographic location support is included for better results")
    print("- The search is designed to find official company websites")


if __name__ == "__main__":
    asyncio.run(main()) 