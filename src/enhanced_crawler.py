"""
Enhanced Async Crawler Module

This module extends the basic AsyncCrawler to collect and store page content
for analysis, specifically focused on finding annual reports and related documents.

CRITICAL FIX: Proper state management to prevent stale data retention between crawls.
"""

import asyncio
import aiohttp
import nest_asyncio
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
import re
from typing import Dict, List, Set, Optional, Tuple
import json
from datetime import datetime

# Apply nest_asyncio to allow running asyncio in Jupyter Notebook
nest_asyncio.apply()


class EnhancedAsyncCrawler:
    """
    Enhanced web crawler that collects page content and metadata for analysis.
    
    This crawler is specifically designed to find annual reports and related
    financial documents on company websites.
    
    CRITICAL FIX: Implements proper state management to prevent stale data
    retention between different crawl operations.
    """
    
    def __init__(self, user_agent: str = 'FinancialReportCrawler/1.0'):
        """
        Initialize the enhanced crawler.
        
        Args:
            user_agent: User agent string for HTTP requests
        """
        self.user_agent = user_agent
        # These will be reset for each crawl operation
        self.robots_parsers: Dict[str, RobotFileParser] = {}
        self.crawl_data: Dict[str, Dict] = {}
        self.visited_urls: Set[str] = set()
        
        # Annual report keywords for detection
        self.report_keywords = [
            'annual report', 'annual', 'financial report', 'investor relations',
            'financial statements', 'sec filings', '10-k', 'form 10-k',
            'sustainability report', 'esg report', 'integrated report',
            'shareholders', 'investor', 'financial', 'earnings',
            'geschäftsbericht', 'rapport annuel', 'informe anual'  # Multi-language support
        ]
        
        # File extensions that might contain reports
        self.report_extensions = ['.pdf', '.doc', '.docx', '.html', '.htm']
        
        # Year patterns for report detection
        self.year_patterns = [
            r'20(20|21|22|23|24)',  # 2020-2024
            r'(20)(20|21|22|23|24)',  # Spaced versions
        ]
    
    def reset_state(self):
        """
        CRITICAL FIX: Reset all state variables before starting a new crawl.
        
        This prevents stale data from previous crawl operations from affecting
        new crawls, which was a major issue in the original implementation.
        """
        print("🔄 Resetting crawler state for new crawl operation...")
        self.robots_parsers.clear()
        self.crawl_data.clear()
        self.visited_urls.clear()
        print("✅ Crawler state reset complete")
    
    def prepare_for_ai_analysis(self, page_content: str, extracted_links: List[Dict], 
                               company_name: str) -> Dict:
        """
        Prepare page data in the format expected by the AI agent for link identification.
        
        This implements the exact format specified in the project plan AI prompt.
        
        Args:
            page_content: The full HTML content of the page
            extracted_links: List of link dictionaries
            company_name: Name of the company being crawled
            
        Returns:
            Dictionary formatted for AI analysis
        """
        # Convert internal link format to AI-expected format
        ai_links = []
        for link in extracted_links:
            ai_links.append({
                'text': link.get('text', ''),
                'href': link.get('url', '')
            })
        
        return {
            'page_content': page_content,
            'extracted_links': ai_links,
            'company_name': company_name
        }
    
    def process_ai_response(self, ai_response: List[Dict]) -> List[Dict]:
        """
        Process AI response and integrate with crawler's internal data structures.
        
        Args:
            ai_response: List of dictionaries from AI analysis
            
        Returns:
            Processed list of potential reports with enhanced metadata
        """
        processed_reports = []
        
        for report in ai_response:
            processed_report = {
                'url': report.get('url', ''),
                'year': report.get('year', 'unknown'),
                'ai_confidence': report.get('confidence', 0.0),
                'ai_reason': report.get('reason', ''),
                'link_text': report.get('text', ''),
                'is_ai_identified': True,
                'timestamp': datetime.now().isoformat()
            }
            processed_reports.append(processed_report)
        
        return processed_reports

    async def can_fetch(self, session: aiohttp.ClientSession, url: str) -> bool:
        """
        Check if URL can be fetched according to robots.txt.
        
        Args:
            session: aiohttp session
            url: URL to check
            
        Returns:
            True if URL can be fetched
        """
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

            if robots_url not in self.robots_parsers:
                parser = RobotFileParser()
                try:
                    async with session.get(robots_url, timeout=5) as response:
                        if response.status == 200:
                            text = await response.text()
                            parser.parse(text.splitlines())
                        else:
                            parser.allow_all = True
                except Exception:
                    parser.allow_all = True
                self.robots_parsers[robots_url] = parser

            parser = self.robots_parsers[robots_url]
            return parser.can_fetch(self.user_agent, url)
        except Exception:
            return True
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """
        Extract clean text content from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Clean text content
        """
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text and clean it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """
        Extract and categorize links from HTML.
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
            
        Returns:
            List of link dictionaries with metadata
        """
        links = []
        base_domain = urlparse(base_url).netloc
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            
            # Only process links within the same domain
            if urlparse(full_url).netloc == base_domain:
                link_text = a_tag.get_text(strip=True).lower()
                
                # Calculate relevance score for annual reports
                relevance_score = self._calculate_link_relevance(link_text, full_url)
                
                # Extract potential year from link text and URL
                year = self._extract_year(link_text + ' ' + full_url)
                
                link_info = {
                    'url': full_url,
                    'text': link_text,
                    'title': a_tag.get('title', ''),
                    'relevance_score': relevance_score,
                    'year': year,
                    'is_potential_report': relevance_score > 0,
                    'is_pdf': full_url.lower().endswith('.pdf')
                }
                
                links.append(link_info)
        
        return links
    
    def _extract_year(self, text: str) -> Optional[str]:
        """
        Extract year from text if it's in our target range (2020-2024).
        
        Args:
            text: Text to search for years
            
        Returns:
            Extracted year or None
        """
        for pattern in self.year_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    year = ''.join(match)
                else:
                    year = match
                if year in ['2020', '2021', '2022', '2023', '2024']:
                    return year
        return None
    
    def _calculate_link_relevance(self, link_text: str, url: str) -> float:
        """
        Calculate relevance score for potential annual report links.
        
        Args:
            link_text: Text content of the link
            url: URL of the link
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0
        text_lower = link_text.lower()
        url_lower = url.lower()
        combined_text = f"{text_lower} {url_lower}"
        
        # High-value keywords
        high_value_keywords = [
            'annual report', 'financial report', 'form 10-k', '10-k',
            'investor relations', 'financial statements'
        ]
        
        for keyword in high_value_keywords:
            if keyword in combined_text:
                score += 0.3
        
        # Medium-value keywords
        medium_value_keywords = [
            'annual', 'financial', 'investor', 'shareholders',
            'sustainability', 'esg', 'integrated'
        ]
        
        for keyword in medium_value_keywords:
            if keyword in combined_text:
                score += 0.1
        
        # Year bonus (2020-2024)
        year_found = self._extract_year(combined_text)
        if year_found:
            score += 0.2
        
        # PDF bonus
        if url_lower.endswith('.pdf'):
            score += 0.2
        
        # Path-based scoring
        if any(path in url_lower for path in ['investor', 'financial', 'annual', 'report']):
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _is_document_url(self, url: str) -> bool:
        """
        Check if URL points to a document.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL appears to be a document
        """
        return any(url.lower().endswith(ext) for ext in self.report_extensions)
    
    async def crawl_page(self, session: aiohttp.ClientSession, url: str) -> Optional[Tuple[str, Dict]]:
        """
        Crawl a single page and extract content and links.
        
        Args:
            session: aiohttp session
            url: URL to crawl
            
        Returns:
            Tuple of (url, page_data) or None if failed
        """
        try:
            # Check robots.txt
            if not await self.can_fetch(session, url):
                return None
            
            # Make request
            headers = {'User-Agent': self.user_agent}
            async with session.get(url, headers=headers, timeout=30) as response:
                if response.status != 200:
                    return None
                
                content_type = response.headers.get('Content-Type', '').lower()
                
                # Handle documents directly
                if self._is_document_url(url) or 'application/pdf' in content_type:
                    return url, {
                        'status': 'document',
                        'content_type': content_type,
                        'size': response.headers.get('Content-Length', 'unknown'),
                        'is_potential_report': True,
                        'timestamp': datetime.now().isoformat()
                    }
                
                # Handle HTML pages
                if 'text/html' in content_type:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    
                    # Extract metadata
                    title = soup.title.string if soup.title else ''
                    text_content = self._extract_text_content(soup)
                    links = self._extract_links(soup, url)
                    
                    # Filter potential report links
                    potential_reports = [link for link in links if link['is_potential_report']]
                    
                    page_data = {
                        'status': 'crawled',
                        'title': title.strip(),
                        'content': text_content[:5000],  # Limit content size
                        'html_content': html_content,  # Store full HTML for AI analysis
                        'links': links,
                        'potential_reports': potential_reports,
                        'link_count': len(links),
                        'potential_report_count': len(potential_reports),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    return url, page_data
                
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return url, {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    async def worker(self, name: str, queue: asyncio.Queue, session: aiohttp.ClientSession):
        """
        Worker function to process URLs from the queue.
        
        Args:
            name: Worker name for logging
            queue: Queue containing URLs to process
            session: aiohttp session
        """
        while True:
            try:
                url = await queue.get()
                
                if url in self.visited_urls:
                    queue.task_done()
                    continue
                
                self.visited_urls.add(url)
                
                result = await self.crawl_page(session, url)
                if result:
                    crawled_url, page_data = result
                    self.crawl_data[crawled_url] = page_data
                    
                    # Add new URLs to queue (only for HTML pages)
                    if page_data.get('status') == 'crawled':
                        base_domain = urlparse(url).netloc
                        for link in page_data.get('links', []):
                            link_url = link['url']
                            if (urlparse(link_url).netloc == base_domain and 
                                link_url not in self.visited_urls and
                                not self._is_document_url(link_url)):
                                queue.put_nowait(link_url)
                    
                    print(f"{name}: ✅ {url} ({page_data.get('status', 'unknown')})")
                else:
                    print(f"{name}: ❌ Failed to crawl {url}")
                    
            except Exception as e:
                print(f"Worker {name}: Error crawling {url}: {e}")
                self.crawl_data[url] = {
                    "status": "error",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            finally:
                queue.task_done()
    
    async def crawl(self, start_url: str, max_pages: int = 100, n_workers: int = 10) -> Dict[str, Dict]:
        """
        Main crawling function with proper state management.
        
        CRITICAL FIX: Resets state before each crawl to prevent stale data retention.
        
        Args:
            start_url: Starting URL for crawling
            max_pages: Maximum number of pages to crawl
            n_workers: Number of concurrent workers
            
        Returns:
            Dictionary containing all crawled data
        """
        # CRITICAL FIX: Reset state before each crawl
        self.reset_state()
        
        print(f"🚀 Starting enhanced crawl of: {start_url}")
        print(f"📊 Configuration: max_pages={max_pages}, workers={n_workers}")
        
        queue = asyncio.Queue()
        queue.put_nowait(start_url)
        
        async with aiohttp.ClientSession() as session:
            # Create worker tasks
            tasks = []
            for i in range(n_workers):
                task = asyncio.create_task(self.worker(f'worker-{i+1}', queue, session))
                tasks.append(task)
            
            # Process until max_pages reached or queue empty
            while len(self.visited_urls) < max_pages:
                try:
                    await asyncio.wait_for(queue.join(), timeout=10.0)
                    if queue.empty():
                        break
                except asyncio.TimeoutError:
                    print("Timeout reached, stopping crawl.")
                    break
            
            # Cancel worker tasks
            for task in tasks:
                task.cancel()
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        print(f"\n✅ Crawling finished. Processed {len(self.visited_urls)} pages.")
        print(f"📈 Found {len(self.get_potential_reports())} potential reports")
        return self.crawl_data
    
    def get_potential_reports(self) -> List[Dict[str, any]]:
        """
        Get all potential annual report links found during crawling.
        
        Returns:
            List of potential report links with metadata
        """
        potential_reports = []
        
        for url, data in self.crawl_data.items():
            if data.get('status') == 'crawled' and 'potential_reports' in data:
                for report_link in data['potential_reports']:
                    potential_reports.append({
                        'source_page': url,
                        'report_url': report_link['url'],
                        'link_text': report_link['text'],
                        'relevance_score': report_link['relevance_score'],
                        'year': report_link.get('year'),
                        'is_pdf': report_link.get('is_pdf', False),
                        'source_title': data.get('title', '')
                    })
            
            elif data.get('status') == 'document' and data.get('is_potential_report'):
                potential_reports.append({
                    'source_page': url,
                    'report_url': url,
                    'link_text': 'Direct document link',
                    'relevance_score': 0.8,
                    'content_type': data.get('content_type', ''),
                    'size': data.get('size', 'unknown'),
                    'is_pdf': 'pdf' in data.get('content_type', '').lower()
                })
        
        # Sort by relevance score
        potential_reports.sort(key=lambda x: x['relevance_score'], reverse=True)
        return potential_reports
    
    def get_pages_for_ai_analysis(self, company_name: str) -> List[Dict]:
        """
        Get pages formatted for AI analysis according to the project plan specification.
        
        Args:
            company_name: Name of the company being analyzed
            
        Returns:
            List of pages formatted for AI analysis
        """
        ai_ready_pages = []
        
        for url, data in self.crawl_data.items():
            if data.get('status') == 'crawled' and 'html_content' in data:
                ai_data = self.prepare_for_ai_analysis(
                    page_content=data['html_content'],
                    extracted_links=data.get('links', []),
                    company_name=company_name
                )
                ai_data['source_url'] = url
                ai_data['page_title'] = data.get('title', '')
                ai_ready_pages.append(ai_data)
        
        return ai_ready_pages
    
    def save_crawl_data(self, filename: str):
        """
        Save crawl data to JSON file.
        
        Args:
            filename: Output filename
        """
        # Create a copy without the full HTML content to reduce file size
        save_data = {}
        for url, data in self.crawl_data.items():
            save_data[url] = {k: v for k, v in data.items() if k != 'html_content'}
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"Crawl data saved to {filename}")


# Example usage
async def main():
    """Test the enhanced crawler with proper state management."""
    crawler = EnhancedAsyncCrawler()
    
    # Test with a company website
    start_url = "https://www.shell.com"
    
    print(f"Starting enhanced crawl of: {start_url}")
    crawl_data = await crawler.crawl(start_url, max_pages=50, n_workers=5)
    
    # Get potential reports
    reports = crawler.get_potential_reports()
    
    print(f"\nFound {len(reports)} potential annual reports:")
    for i, report in enumerate(reports[:10]):  # Show top 10
        print(f"{i+1}. {report['link_text']} (Score: {report['relevance_score']:.2f})")
        print(f"   URL: {report['report_url']}")
        if report.get('year'):
            print(f"   Year: {report['year']}")
        print()
    
    # Test state reset by running another crawl
    print("\n" + "="*60)
    print("Testing state reset with another crawl...")
    
    crawl_data_2 = await crawler.crawl("https://www.microsoft.com", max_pages=30, n_workers=3)
    reports_2 = crawler.get_potential_reports()
    
    print(f"Second crawl found {len(reports_2)} potential reports")
    print("✅ State management test complete")
    
    # Save data
    crawler.save_crawl_data("crawl_results.json")


if __name__ == "__main__":
    asyncio.run(main())
