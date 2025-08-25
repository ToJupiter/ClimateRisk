"""
Website Finder Module

This module is responsible for finding company websites using search APIs.
It takes company names and returns their official website URLs.
"""

import asyncio
import aiohttp
from typing import List, Dict, Optional
from urllib.parse import urlparse
import re


class WebsiteFinder:
    """
    A class to find company websites using search engines.
    
    This class uses multiple search strategies to find the official website
    of companies based on their names.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the WebsiteFinder.
        
        Args:
            api_key: Optional API key for search services (e.g., Google Custom Search)
        """
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def find_company_website(self, company_name: str, country_code: str = "") -> Optional[str]:
        """
        Find the official website for a given company.
        
        Args:
            company_name: Name of the company to search for
            country_code: Optional country code to improve search accuracy
            
        Returns:
            The official website URL if found, None otherwise
        """
        try:
            # Clean company name for search
            clean_name = self._clean_company_name(company_name)
            
            # Try different search strategies
            website = await self._search_duckduckgo(clean_name, country_code)
            
            if website:
                return self._validate_and_clean_url(website)
            
            return None
            
        except Exception as e:
            print(f"Error finding website for {company_name}: {e}")
            return None
    
    def _clean_company_name(self, company_name: str) -> str:
        """
        Clean company name for better search results.
        
        Args:
            company_name: Raw company name
            
        Returns:
            Cleaned company name suitable for search
        """
        # Remove common suffixes and patterns
        suffixes = ['Ltd', 'Limited', 'Inc', 'Corporation', 'Corp', 'SA', 'AS', 'PCL', 
                   'Bhd', 'Tbk', 'PT', 'PLC', 'Nyrt', 'SAA', 'BSC', 'SAOG', 'ESP']
        
        # Remove parentheses and their contents
        clean_name = re.sub(r'\([^)]*\)', '', company_name)
        
        # Remove common suffixes
        for suffix in suffixes:
            clean_name = re.sub(rf'\b{suffix}\b', '', clean_name, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        clean_name = ' '.join(clean_name.split())
        
        return clean_name.strip()
    
    async def _search_duckduckgo(self, company_name: str, country_code: str = "") -> Optional[str]:
        """
        Search for company website using DuckDuckGo.
        
        Args:
            company_name: Cleaned company name
            country_code: Optional country code
            
        Returns:
            Website URL if found
        """
        try:
            # Construct search query
            query = f"{company_name} official website"
            if country_code:
                query += f" {country_code}"
            
            # DuckDuckGo instant answer API
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract website from abstract or infobox
                    if 'AbstractURL' in data and data['AbstractURL']:
                        return data['AbstractURL']
                    
                    # Check related topics
                    if 'RelatedTopics' in data:
                        for topic in data['RelatedTopics']:
                            if 'FirstURL' in topic and topic['FirstURL']:
                                url = topic['FirstURL']
                                if self._is_likely_company_website(url, company_name):
                                    return url
            
            return None
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            return None
    
    def _is_likely_company_website(self, url: str, company_name: str) -> bool:
        """
        Check if URL is likely the company's official website.
        
        Args:
            url: URL to check
            company_name: Company name for comparison
            
        Returns:
            True if URL is likely the official website
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove common prefixes
            domain = re.sub(r'^www\.', '', domain)
            
            # Check if company name appears in domain
            company_words = company_name.lower().split()
            for word in company_words:
                if len(word) > 3 and word in domain:
                    return True
            
            # Avoid social media and other non-official sites
            excluded_domains = ['wikipedia', 'linkedin', 'facebook', 'twitter', 
                              'bloomberg', 'reuters', 'google', 'yahoo']
            
            for excluded in excluded_domains:
                if excluded in domain:
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_and_clean_url(self, url: str) -> Optional[str]:
        """
        Validate and clean the found URL.
        
        Args:
            url: URL to validate
            
        Returns:
            Cleaned and validated URL
        """
        try:
            if not url:
                return None
            
            # Add protocol if missing
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Parse and validate
            parsed = urlparse(url)
            if not parsed.netloc:
                return None
            
            # Return clean URL
            return f"{parsed.scheme}://{parsed.netloc}"
            
        except Exception:
            return None
    
    async def find_multiple_websites(self, companies: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
        """
        Find websites for multiple companies concurrently.
        
        Args:
            companies: List of dictionaries with 'name' and optionally 'country' keys
            
        Returns:
            Dictionary mapping company names to their website URLs
        """
        tasks = []
        for company in companies:
            name = company.get('name', '')
            country = company.get('country', '')
            if name:
                task = self.find_company_website(name, country)
                tasks.append((name, task))
        
        results = {}
        for name, task in tasks:
            try:
                website = await task
                results[name] = website
                print(f"Found website for {name}: {website}")
            except Exception as e:
                print(f"Error finding website for {name}: {e}")
                results[name] = None
        
        return results


# Example usage and testing
async def main():
    """Test the WebsiteFinder with sample companies."""
    test_companies = [
        {"name": "London Stock Exchange Group PLC", "country": "GB"},
        {"name": "Arcelik AS", "country": "TR"},
        {"name": "Delta Electronics Thailand PCL", "country": "TH"}
    ]
    
    async with WebsiteFinder() as finder:
        websites = await finder.find_multiple_websites(test_companies)
        
        print("\n--- Found Websites ---")
        for company, website in websites.items():
            print(f"{company}: {website}")


if __name__ == "__main__":
    asyncio.run(main())
