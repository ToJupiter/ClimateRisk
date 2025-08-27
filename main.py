"""
Annual Report Crawler - Main Orchestrator

This is the main entry point for the annual report crawler system.
It coordinates all components to find and download annual reports for companies.

Usage:
    python main.py --config config.json
    python main.py --company "Apple Inc" --openai-key "your-key"
    python main.py --csv data/2020_ESG-SCORES-28-COUNTRIES.csv
"""

import asyncio
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from website_finder import WebsiteFinder
from enhanced_crawler import EnhancedAsyncCrawler
from report_finder import BM25ReportFinder, ReportAnalyzer
from openai_selector import OpenAIReportSelector, ReportCandidate
from report_downloader import ReportDownloader, prepare_reports_for_download
from ai_link_analyzer import AILinkAnalyzer


class AnnualReportCrawlerOrchestrator:
    """
    Main orchestrator class that coordinates all components of the annual report crawler.
    
    This class manages the entire pipeline from finding company websites to
    downloading annual reports.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the orchestrator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.website_finder = None
        self.crawler = None
        self.report_finder = None
        self.openai_selector = None
        self.ai_link_analyzer = None
        self.downloader = None
        
        # Results storage
        self.results = {
            'companies_processed': [],
            'websites_found': {},
            'crawl_results': {},
            'bm25_results': {},
            'ai_selected_reports': {},
            'download_results': {},
            'summary': {}
        }
    
    def _initialize_components(self):
        """Initialize all components with configuration."""
        # Initialize website finder with OpenAI API key
        openai_config = self.config.get('openai', {})
        api_key = openai_config.get('api_key')
        self.website_finder = WebsiteFinder(api_key=api_key)
        
        # Initialize crawler
        crawler_config = self.config.get('crawler', {})
        user_agent = crawler_config.get('user_agent', 'FinancialReportCrawler/1.0')
        self.crawler = EnhancedAsyncCrawler(user_agent=user_agent)
        
        # Initialize BM25 report finder
        bm25_config = self.config.get('bm25', {})
        k1 = bm25_config.get('k1', 1.2)
        b = bm25_config.get('b', 0.75)
        self.report_finder = BM25ReportFinder(k1=k1, b=b)
        
        # Initialize OpenAI selector (if API key provided)
        openai_config = self.config.get('openai', {})
        api_key = openai_config.get('api_key')
        if api_key:
            model = openai_config.get('model', 'gpt-3.5-turbo')
            self.openai_selector = OpenAIReportSelector(api_key=api_key, model=model)
            # Also initialize the AI link analyzer
            self.ai_link_analyzer = AILinkAnalyzer(api_key=api_key, model=model)
        
        # Initialize downloader
        download_config = self.config.get('download', {})
        output_dir = download_config.get('output_dir', 'output_data')
        max_file_size = download_config.get('max_file_size', 100 * 1024 * 1024)
        self.downloader = ReportDownloader(output_dir=output_dir, max_file_size=max_file_size)
    
    async def process_single_company(self, company_name: str, 
                                   country_code: str = "") -> Dict:
        """
        Process a single company to find and download its annual reports.
        
        Args:
            company_name: Name of the company
            country_code: Optional country code
            
        Returns:
            Processing results for the company
        """
        print(f"\n{'='*60}")
        print(f"Processing company: {company_name}")
        print(f"{'='*60}")
        
        company_results = {
            'company_name': company_name,
            'country_code': country_code,
            'timestamp': datetime.now().isoformat(),
            'status': 'started'
        }
        
        try:
            # Step 1: Find company website
            print(f"\nStep 1: Finding website for {company_name}")
            async with WebsiteFinder() as finder:
                website = await finder.find_company_website(company_name, country_code)
            
            if not website:
                company_results['status'] = 'failed'
                company_results['error'] = 'Website not found'
                print(f"❌ Could not find website for {company_name}")
                return company_results
            
            print(f"✅ Found website: {website}")
            company_results['website'] = website
            
            # Step 2: Crawl website
            print(f"\nStep 2: Crawling website {website}")
            crawler_config = self.config.get('crawler', {})
            max_pages = crawler_config.get('max_pages', 100)
            n_workers = crawler_config.get('n_workers', 10)
            
            crawl_data = await self.crawler.crawl(
                start_url=website, 
                max_pages=max_pages, 
                n_workers=n_workers
            )
            
            if not crawl_data:
                company_results['status'] = 'failed'
                company_results['error'] = 'No crawl data collected'
                print(f"❌ Failed to crawl {website}")
                return company_results
            
            print(f"✅ Crawled {len(crawl_data)} pages")
            company_results['crawl_stats'] = {
                'pages_crawled': len(crawl_data),
                'potential_reports_found': len(self.crawler.get_potential_reports())
            }
            
            # Step 3: Use BM25 to find potential reports
            print(f"\nStep 3: Analyzing crawled data with BM25")
            search_results = self.report_finder.search_reports(crawl_data)
            best_links = self.report_finder.extract_best_report_links(search_results)
            
            if not best_links:
                company_results['status'] = 'completed'
                company_results['warning'] = 'No potential reports found by BM25'
                print(f"⚠️ No potential reports found for {company_name}")
                return company_results
            
            print(f"✅ Found {len(best_links)} potential report links")
            
            # Step 4: Use AI to analyze pages and identify reports (if available)
            if self.ai_link_analyzer:
                print(f"\nStep 4a: Using AI to analyze crawled pages for report links")
                
                # Get pages formatted for AI analysis
                ai_ready_pages = self.crawler.get_pages_for_ai_analysis(company_name)
                
                if ai_ready_pages:
                    # Limit pages for API cost control
                    max_pages_for_ai = self.config.get('ai_analysis', {}).get('max_pages', 20)
                    pages_to_analyze = ai_ready_pages[:max_pages_for_ai]
                    
                    # Perform AI analysis
                    ai_analysis_results = await self.ai_link_analyzer.analyze_multiple_pages(
                        pages_to_analyze, company_name
                    )
                    
                    # Aggregate AI results
                    ai_reports = self.ai_link_analyzer.aggregate_results(ai_analysis_results)
                    
                    if ai_reports:
                        print(f"✅ AI analysis found {len(ai_reports)} high-confidence reports")
                        
                        # Convert AI reports to the expected format and combine with BM25 results
                        combined_reports = []
                        
                        # Add AI-identified reports with high priority
                        for ai_report in ai_reports:
                            if ai_report['confidence'] >= 0.7:  # High confidence threshold
                                combined_report = {
                                    'url': ai_report['url'],
                                    'link_text': ai_report.get('link_text', 'AI-identified'),
                                    'source_page': ai_report['source_page'],
                                    'source_title': ai_report.get('source_title', ''),
                                    'year': ai_report['year'],
                                    'bm25_score': 0.0,  # AI reports don't have BM25 scores
                                    'relevance_score': ai_report['confidence'],
                                    'combined_score': ai_report['confidence'] + 0.2,  # Boost for AI identification
                                    'ai_confidence': ai_report['confidence'],
                                    'ai_reason': ai_report['reason'],
                                    'is_ai_identified': True
                                }
                                combined_reports.append(combined_report)
                        
                        # Add best BM25 results that aren't already identified by AI
                        ai_urls = {report['url'] for report in ai_reports}
                        for bm25_report in best_links:
                            if bm25_report['url'] not in ai_urls:
                                bm25_report['is_ai_identified'] = False
                                combined_reports.append(bm25_report)
                        
                        best_links = combined_reports
                        print(f"✅ Combined analysis: {len(best_links)} total potential reports")
                    else:
                        print(f"⚠️ AI analysis didn't find additional high-confidence reports")
                else:
                    print(f"⚠️ No pages available for AI analysis")
            
            # Step 4b: Use OpenAI selector to choose best reports by year (if available)
            if self.openai_selector:
                print(f"\nStep 4b: Using AI to select best reports by year")
                
                # Convert to ReportCandidate objects
                candidates = []
                for link in best_links[:30]:  # Increased limit since we have better filtering
                    candidate = ReportCandidate(
                        url=link['url'],
                        link_text=link['link_text'],
                        source_page=link['source_page'],
                        source_title=link['source_title'],
                        year=link['year'],
                        bm25_score=link.get('bm25_score', 0.0),
                        relevance_score=link['relevance_score'],
                        confidence_score=link['combined_score']
                    )
                    candidates.append(candidate)
                
                # Get AI-selected reports by year
                target_years = self.config.get('target_years', ['2020', '2021', '2022', '2023', '2024'])
                selected_reports = await self.openai_selector.select_best_reports_by_year(
                    candidates, company_name, target_years
                )
                
                # Count selected reports
                total_selected = sum(len(reports) for reports in selected_reports.values())
                print(f"✅ AI selected {total_selected} reports across {len(selected_reports)} years")
                
            else:
                print(f"\nStep 4: Skipping AI selection (no OpenAI API key)")
                # Use rule-based selection
                analyzer = ReportAnalyzer()
                filtered_by_year = analyzer.filter_by_year(best_links)
                selected_reports = {
                    year: reports[:3] for year, reports in filtered_by_year.items()
                }
            
            company_results['reports_found'] = {
                year: len(reports) for year, reports in selected_reports.items()
            }
            
            # Step 5: Download reports
            print(f"\nStep 5: Downloading reports")
            download_list = prepare_reports_for_download(selected_reports, company_name)
            
            if not download_list:
                company_results['status'] = 'completed'
                company_results['warning'] = 'No reports selected for download'
                print(f"⚠️ No reports to download for {company_name}")
                return company_results
            
            download_config = self.config.get('download', {})
            max_concurrent = download_config.get('max_concurrent', 5)
            
            download_results = await self.downloader.download_multiple_reports(
                download_list, max_concurrent=max_concurrent
            )
            
            # Count successful downloads
            successful_downloads = sum(1 for r in download_results if r['status'] == 'success')
            print(f"✅ Successfully downloaded {successful_downloads} reports")
            
            company_results['status'] = 'completed'
            company_results['downloads'] = {
                'attempted': len(download_results),
                'successful': successful_downloads,
                'failed': len(download_results) - successful_downloads
            }
            
            return company_results
            
        except Exception as e:
            print(f"❌ Error processing {company_name}: {str(e)}")
            company_results['status'] = 'failed'
            company_results['error'] = str(e)
            return company_results
    
    async def process_companies_from_csv(self, csv_file: str) -> Dict:
        """
        Process companies from a CSV file.
        
        Args:
            csv_file: Path to CSV file with company data
            
        Returns:
            Overall processing results
        """
        print(f"Processing companies from CSV: {csv_file}")
        
        companies = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    company_name = row.get('Company Name', '').strip()
                    country_code = row.get('HQ', '').strip()
                    
                    if company_name:
                        companies.append({
                            'name': company_name,
                            'country': country_code
                        })
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return {'error': str(e)}
        
        print(f"Found {len(companies)} companies in CSV")
        
        # Process companies with optional limiting
        max_companies = self.config.get('max_companies')
        if max_companies:
            companies = companies[:max_companies]
            print(f"Processing limited to {max_companies} companies")
        
        # Process each company
        results = []
        for i, company in enumerate(companies, 1):
            print(f"\n{'='*20} Company {i}/{len(companies)} {'='*20}")
            
            result = await self.process_single_company(
                company['name'], 
                company['country']
            )
            results.append(result)
            
            # Optional delay between companies
            delay = self.config.get('delay_between_companies', 0)
            if delay > 0:
                print(f"Waiting {delay} seconds before next company...")
                await asyncio.sleep(delay)
        
        return {
            'total_companies': len(companies),
            'results': results,
            'summary': self._create_summary(results)
        }
    
    def _create_summary(self, results: List[Dict]) -> Dict:
        """
        Create a summary of processing results.
        
        Args:
            results: List of company processing results
            
        Returns:
            Summary statistics
        """
        total = len(results)
        completed = sum(1 for r in results if r.get('status') == 'completed')
        failed = sum(1 for r in results if r.get('status') == 'failed')
        
        total_downloads = 0
        successful_downloads = 0
        
        for result in results:
            downloads = result.get('downloads', {})
            total_downloads += downloads.get('attempted', 0)
            successful_downloads += downloads.get('successful', 0)
        
        return {
            'total_companies': total,
            'completed_successfully': completed,
            'failed': failed,
            'success_rate': f"{completed/total*100:.1f}%" if total > 0 else "0%",
            'total_downloads_attempted': total_downloads,
            'successful_downloads': successful_downloads,
            'download_success_rate': f"{successful_downloads/total_downloads*100:.1f}%" if total_downloads > 0 else "0%"
        }
    
    async def run(self, companies: Optional[List[Dict]] = None, 
                 csv_file: Optional[str] = None) -> Dict:
        """
        Main execution method.
        
        Args:
            companies: Optional list of companies to process
            csv_file: Optional CSV file path
            
        Returns:
            Execution results
        """
        print("🚀 Starting Annual Report Crawler")
        print(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Initialize components
        self._initialize_components()
        
        start_time = datetime.now()
        
        try:
            if csv_file:
                results = await self.process_companies_from_csv(csv_file)
            elif companies:
                results = []
                for company in companies:
                    result = await self.process_single_company(
                        company.get('name', ''),
                        company.get('country', '')
                    )
                    results.append(result)
                results = {
                    'total_companies': len(companies),
                    'results': results,
                    'summary': self._create_summary(results)
                }
            else:
                raise ValueError("Either companies list or csv_file must be provided")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"crawler_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"\n{'='*60}")
            print(f"🎉 Crawling completed in {duration:.2f} seconds")
            print(f"📊 Summary: {results.get('summary', {})}")
            print(f"💾 Results saved to: {results_file}")
            print(f"{'='*60}")
            
            return results
            
        except Exception as e:
            print(f"❌ Crawler failed: {str(e)}")
            raise


def load_config(config_file: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Using default configuration.")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return get_default_config()


def get_default_config() -> Dict:
    """Get default configuration."""
    return {
        "target_years": ["2020", "2021", "2022", "2023", "2024"],
        "crawler": {
            "max_pages": 100,
            "n_workers": 10,
            "user_agent": "FinancialReportCrawler/1.0"
        },
        "bm25": {
            "k1": 1.2,
            "b": 0.75
        },
        "openai": {
            "api_key": None,
            "model": "gpt-3.5-turbo"
        },
        "ai_analysis": {
            "max_pages": 20,
            "confidence_threshold": 0.7
        },
        "download": {
            "output_dir": "output_data",
            "max_file_size": 104857600,
            "max_concurrent": 5
        },
        "max_companies": None,
        "delay_between_companies": 1
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Annual Report Crawler")
    parser.add_argument('--config', help='Configuration file path', default='config.json')
    parser.add_argument('--csv', help='CSV file with companies')
    parser.add_argument('--company', help='Single company name to process')
    parser.add_argument('--country', help='Country code for single company', default='')
    parser.add_argument('--openai-key', help='OpenAI API key')
    parser.add_argument('--max-companies', type=int, help='Maximum number of companies to process')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.openai_key:
        config['openai']['api_key'] = args.openai_key
    
    if args.max_companies:
        config['max_companies'] = args.max_companies
    
    # Create orchestrator
    orchestrator = AnnualReportCrawlerOrchestrator(config)
    
    # Run crawler
    try:
        if args.csv:
            asyncio.run(orchestrator.run(csv_file=args.csv))
        elif args.company:
            companies = [{'name': args.company, 'country': args.country}]
            asyncio.run(orchestrator.run(companies=companies))
        else:
            # Default: use the ESG CSV file
            default_csv = "data/2020_ESG-SCORES-28-COUNTRIES.csv"
            if Path(default_csv).exists():
                asyncio.run(orchestrator.run(csv_file=default_csv))
            else:
                print(f"No input specified and default CSV {default_csv} not found.")
                print("Use --csv, --company, or --help for options.")
    except KeyboardInterrupt:
        print("\n⚠️ Crawler interrupted by user")
    except Exception as e:
        print(f"❌ Crawler error: {e}")
        raise


if __name__ == "__main__":
    main()
