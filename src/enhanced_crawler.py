"""
Enhanced Async Crawler Module

This module extends the basic AsyncCrawler to collect and store page content
for analysis, specifically focused on finding annual reports and related documents.
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
    """
    
    def __init__(self, user_agent: str = 'FinancialReportCrawler/1.0'):
        """
        Initialize the enhanced crawler.
        
        Args:
            user_agent: User agent string for HTTP requests
        """
        self.user_agent = user_agent
        self.robots_parsers: Dict[str, RobotFileParser] = {}
        self.crawl_data: Dict[str, Dict] = {}
        self.visited_urls: Set[str] = set()
        
        # Annual report keywords for detection
        self.report_keywords = [
            'annual report', 'annual', 'financial report', 'investor relations',
            'financial statements', 'sec filings', '10-k', 'form 10-k',
            'sustainability report', 'esg report', 'integrated report',
            'shareholders', 'investor', 'financial', 'earnings'
        ]
        
        # File extensions that might contain reports
        self.report_extensions = ['.pdf', '.doc', '.docx', '.html', '.htm']
    
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
                
                link_info = {
                    'url': full_url,
                    'text': link_text,
                    'title': a_tag.get('title', ''),
                    'relevance_score': relevance_score,
                    'is_potential_report': relevance_score > 0
                }
                
                links.append(link_info)
        
        return links
    
    def _calculate_link_relevance(self, link_text: str, url: str) -> float:
        """
        Calculate relevance score for potential annual report links.
        
        Args:
            link_text: Text content of the link
            url: URL of the link
            
        Returns:
            Relevance score (0-1, higher is more relevant)
        """
        score = 0.0
        text_lower = link_text.lower()
        url_lower = url.lower()
        
        # Check for year patterns (2020-2024)
        year_pattern = r'\b(202[0-4])\b'
        if re.search(year_pattern, text_lower) or re.search(year_pattern, url_lower):
            score += 0.3
        
        # Check for report keywords
        for keyword in self.report_keywords:
            if keyword in text_lower:
                score += 0.2
            if keyword in url_lower:
                score += 0.1
        
        # Check for file extensions
        for ext in self.report_extensions:
            if url_lower.endswith(ext):
                score += 0.2
                break
        
        # Boost score for investor relations sections
        investor_terms = ['investor', 'financial', 'report', 'sec', 'filing']
        for term in investor_terms:
            if term in url_lower:
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Extract page metadata.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text(strip=True)
        
        # Meta description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag:
            metadata['description'] = desc_tag.get('content', '')
        
        # Meta keywords
        keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            metadata['keywords'] = keywords_tag.get('content', '')
        
        return metadata
    
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
                
                # Check robots.txt
                if not await self.can_fetch(session, url):
                    print(f"Worker {name}: Skipped (robots.txt): {url}")
                    self.crawl_data[url] = {"status": "skipped_robots", "timestamp": datetime.now().isoformat()}
                    queue.task_done()
                    continue
                
                print(f"Worker {name}: Crawling {url}")
                
                headers = {'User-Agent': self.user_agent}
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        
                        if 'text/html' in content_type:
                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')
                            
                            # Extract page data
                            text_content = self._extract_text_content(soup)
                            links = self._extract_links(soup, url)
                            metadata = self._extract_metadata(soup)
                            
                            # Store comprehensive page data
                            self.crawl_data[url] = {
                                "status": "crawled",
                                "timestamp": datetime.now().isoformat(),
                                "content_type": content_type,
                                "title": metadata.get('title', ''),
                                "description": metadata.get('description', ''),
                                "keywords": metadata.get('keywords', ''),
                                "text_content": text_content[:5000],  # Limit text length
                                "links": links,
                                "potential_reports": [link for link in links if link['is_potential_report']],
                                "word_count": len(text_content.split())
                            }
                            
                            # Add high-relevance links to queue
                            for link in links:
                                if link['relevance_score'] > 0.5 and link['url'] not in self.visited_urls:
                                    await queue.put(link['url'])
                        
                        elif any(ext in content_type for ext in ['pdf', 'doc', 'application']):
                            # Handle potential document files
                            self.crawl_data[url] = {
                                "status": "document",
                                "timestamp": datetime.now().isoformat(),
                                "content_type": content_type,
                                "size": response.headers.get('content-length', 'unknown'),
                                "is_potential_report": True
                            }
                        
                        else:
                            self.crawl_data[url] = {
                                "status": "unsupported_content",
                                "timestamp": datetime.now().isoformat(),
                                "content_type": content_type
                            }
                    
                    else:
                        self.crawl_data[url] = {
                            "status": f"failed_status_{response.status}",
                            "timestamp": datetime.now().isoformat()
                        }
            
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
        Main crawling function.
        
        Args:
            start_url: Starting URL for crawling
            max_pages: Maximum number of pages to crawl
            n_workers: Number of concurrent workers
            
        Returns:
            Dictionary containing all crawled data
        """
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
        
        print(f"\nCrawling finished. Processed {len(self.visited_urls)} pages.")
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
                        'source_title': data.get('title', '')
                    })
            
            elif data.get('status') == 'document' and data.get('is_potential_report'):
                potential_reports.append({
                    'source_page': url,
                    'report_url': url,
                    'link_text': 'Direct document link',
                    'relevance_score': 0.8,
                    'content_type': data.get('content_type', ''),
                    'size': data.get('size', 'unknown')
                })
        
        # Sort by relevance score
        potential_reports.sort(key=lambda x: x['relevance_score'], reverse=True)
        return potential_reports
    
    def save_crawl_data(self, filename: str):
        """
        Save crawl data to JSON file.
        
        Args:
            filename: Output filename
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.crawl_data, f, indent=2, ensure_ascii=False)
        print(f"Crawl data saved to {filename}")


# Example usage
async def main():
    """Test the enhanced crawler."""
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
        print()
    
    # Save data
    crawler.save_crawl_data("crawl_results.json")


if __name__ == "__main__":
    asyncio.run(main())
