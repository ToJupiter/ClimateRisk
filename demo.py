"""
Demo Script for Annual Report Crawler

This script demonstrates how to use the crawler system with a simple example.
"""

import asyncio
import json
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from website_finder import WebsiteFinder
from enhanced_crawler import EnhancedAsyncCrawler
from report_finder import BM25ReportFinder
from report_downloader import ReportDownloader


async def demo_website_finder():
    """Demo the website finder functionality."""
    print("🔍 Demo: Website Finder")
    print("-" * 40)
    
    test_companies = [
        {"name": "Microsoft Corporation", "country": "US"},
        {"name": "Apple Inc", "country": "US"},
        {"name": "Shell plc", "country": "GB"}
    ]
    
    async with WebsiteFinder() as finder:
        for company in test_companies:
            website = await finder.find_company_website(
                company["name"], 
                company["country"]
            )
            print(f"{company['name']}: {website}")
    
    print()


async def demo_crawler():
    """Demo the crawler functionality."""
    print("🕷️ Demo: Enhanced Crawler")
    print("-" * 40)
    
    # Use a simple test website
    test_url = "https://httpbin.org"  # Simple test site
    
    crawler = EnhancedAsyncCrawler()
    
    print(f"Crawling: {test_url}")
    crawl_data = await crawler.crawl(
        start_url=test_url, 
        max_pages=5, 
        n_workers=2
    )
    
    print(f"Crawled {len(crawl_data)} pages")
    
    # Show potential reports (likely none for test site)
    potential_reports = crawler.get_potential_reports()
    print(f"Found {len(potential_reports)} potential reports")
    
    print()


def demo_bm25():
    """Demo the BM25 report finder."""
    print("🎯 Demo: BM25 Report Finder")
    print("-" * 40)
    
    # Create sample crawl data
    sample_data = {
        "https://example.com/investor": {
            "status": "crawled",
            "title": "Investor Relations - Annual Reports 2023",
            "description": "Find our latest annual reports and financial statements",
            "text_content": "Our annual report for 2023 provides comprehensive financial information and ESG metrics for our stakeholders.",
            "potential_reports": [
                {
                    "url": "https://example.com/annual-report-2023.pdf",
                    "text": "Annual Report 2023",
                    "relevance_score": 0.9
                },
                {
                    "url": "https://example.com/financial-statements-2022.pdf", 
                    "text": "Financial Statements 2022",
                    "relevance_score": 0.7
                }
            ]
        },
        "https://example.com/sustainability": {
            "status": "crawled",
            "title": "Sustainability and ESG Report 2023",
            "description": "Our commitment to environmental and social governance",
            "text_content": "This sustainability report outlines our ESG initiatives and environmental impact for 2023.",
            "potential_reports": [
                {
                    "url": "https://example.com/esg-report-2023.pdf",
                    "text": "ESG Report 2023", 
                    "relevance_score": 0.6
                }
            ]
        }
    }
    
    finder = BM25ReportFinder()
    results = finder.search_reports(sample_data)
    
    print("BM25 Search Results:")
    for query, docs in results.items():
        if docs:  # Only show queries with results
            print(f"\n{query}:")
            for doc in docs[:3]:  # Top 3 results
                print(f"  • {doc['title']} (Score: {doc['score']:.2f})")
    
    # Extract best links
    best_links = finder.extract_best_report_links(results)
    print(f"\nFound {len(best_links)} high-confidence report links")
    
    for link in best_links[:3]:  # Show top 3
        print(f"  • {link['link_text']} (Year: {link.get('year', 'Unknown')})")
        print(f"    URL: {link['url']}")
        print(f"    Combined Score: {link['combined_score']:.2f}")
    
    print()


def demo_downloader():
    """Demo the report downloader (without actually downloading)."""
    print("📥 Demo: Report Downloader")
    print("-" * 40)
    
    downloader = ReportDownloader(output_dir="demo_output")
    
    # Show directory structure
    print("Created directory structure:")
    for root, dirs, files in os.walk("demo_output"):
        level = root.replace("demo_output", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    # Get summary
    summary = downloader.get_download_summary()
    print(f"\nDownload Summary:")
    print(f"  Total files: {summary['total_files']}")
    print(f"  Files by year: {summary['files_by_year']}")
    print(f"  Output directory: {summary['output_directory']}")
    
    print()


async def main():
    """Run all demos."""
    print("🚀 Annual Report Crawler - Demo")
    print("=" * 50)
    print()
    
    # Demo 1: Website Finder
    await demo_website_finder()
    
    # Demo 2: Crawler (commented out to avoid making external requests)
    # await demo_crawler()
    
    # Demo 3: BM25 Report Finder
    demo_bm25()
    
    # Demo 4: Report Downloader
    demo_downloader()
    
    print("✅ Demo completed!")
    print("\nTo run the full crawler:")
    print("  python main.py --company 'Apple Inc' --country US")
    print("  python main.py --csv data/2020_ESG-SCORES-28-COUNTRIES.csv --max-companies 3")


if __name__ == "__main__":
    asyncio.run(main())
