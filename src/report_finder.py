"""
Annual Report Finder Module

This module implements BM25 and other search algorithms to find and rank
annual reports from crawled website data.
"""

import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional
import json
from datetime import datetime


class BM25ReportFinder:
    """
    BM25-based search algorithm specifically tuned for finding annual reports.
    
    This class implements the BM25 ranking function with customizations for
    financial document discovery.
    """
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        """
        Initialize BM25 parameters.
        
        Args:
            k1: Controls term frequency saturation point (default: 1.2)
            b: Controls how much document length normalizes scores (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        
        # Annual report specific query terms with weights
        self.report_queries = {
            'annual_report_2020': {
                'terms': ['annual', 'report', '2020'],
                'weight': 1.0
            },
            'annual_report_2021': {
                'terms': ['annual', 'report', '2021'],
                'weight': 1.0
            },
            'annual_report_2022': {
                'terms': ['annual', 'report', '2022'],
                'weight': 1.0
            },
            'annual_report_2023': {
                'terms': ['annual', 'report', '2023'],
                'weight': 1.0
            },
            'annual_report_2024': {
                'terms': ['annual', 'report', '2024'],
                'weight': 1.0
            },
            'financial_statements': {
                'terms': ['financial', 'statements', 'report'],
                'weight': 0.8
            },
            'investor_relations': {
                'terms': ['investor', 'relations', 'annual'],
                'weight': 0.7
            },
            'sec_filings': {
                'terms': ['sec', 'filing', '10-k', 'form'],
                'weight': 0.9
            },
            'sustainability_report': {
                'terms': ['sustainability', 'esg', 'report'],
                'weight': 0.6
            }
        }
        
        # Document corpus for BM25
        self.documents: List[Dict] = []
        self.doc_frequencies: Dict[str, int] = defaultdict(int)
        self.idf_cache: Dict[str, float] = {}
        self.avg_doc_length: float = 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 processing.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and extract alphanumeric tokens
        text = text.lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', text)
        
        # Filter out very short tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return tokens
    
    def _calculate_idf(self, term: str, doc_count: int) -> float:
        """
        Calculate IDF (Inverse Document Frequency) for a term.
        
        Args:
            term: Term to calculate IDF for
            doc_count: Total number of documents
            
        Returns:
            IDF value
        """
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        df = self.doc_frequencies.get(term, 0)
        if df == 0:
            idf = 0.0
        else:
            idf = math.log((doc_count - df + 0.5) / (df + 0.5))
        
        self.idf_cache[term] = idf
        return idf
    
    def _build_corpus(self, crawl_data: Dict[str, Dict]):
        """
        Build document corpus from crawl data.
        
        Args:
            crawl_data: Crawled website data
        """
        self.documents = []
        self.doc_frequencies = defaultdict(int)
        self.idf_cache = {}
        
        total_length = 0
        
        for url, data in crawl_data.items():
            if data.get('status') != 'crawled':
                continue
            
            # Combine relevant text fields
            text_parts = []
            
            # Add title with higher weight (repeat 3 times)
            title = data.get('title', '')
            text_parts.extend([title] * 3)
            
            # Add description
            description = data.get('description', '')
            text_parts.append(description)
            
            # Add keywords
            keywords = data.get('keywords', '')
            text_parts.append(keywords)
            
            # Add potential report link texts
            potential_reports = data.get('potential_reports', [])
            for report in potential_reports:
                link_text = report.get('text', '')
                # Give link text higher weight (repeat 5 times)
                text_parts.extend([link_text] * 5)
            
            # Add partial page content
            content = data.get('text_content', '')[:2000]  # Limit content
            text_parts.append(content)
            
            # Combine and tokenize
            combined_text = ' '.join(filter(None, text_parts))
            tokens = self._tokenize(combined_text)
            
            if not tokens:
                continue
            
            # Create document record
            doc = {
                'url': url,
                'tokens': tokens,
                'length': len(tokens),
                'title': title,
                'potential_reports': potential_reports,
                'data': data
            }
            
            self.documents.append(doc)
            total_length += len(tokens)
            
            # Update document frequencies
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.doc_frequencies[token] += 1
        
        # Calculate average document length
        if self.documents:
            self.avg_doc_length = total_length / len(self.documents)
    
    def _bm25_score(self, query_terms: List[str], document: Dict[str, any]) -> float:
        """
        Calculate BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            document: Document dictionary
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_tokens = document['tokens']
        doc_length = document['length']
        
        # Count term frequencies in document
        term_counts = Counter(doc_tokens)
        
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            
            # Calculate IDF
            idf = self._calculate_idf(term, len(self.documents))
            
            # Calculate BM25 component for this term
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def _boost_score(self, score: float, document: Dict[str, any], query_name: str) -> float:
        """
        Apply boost factors to BM25 score based on document characteristics.
        
        Args:
            score: Base BM25 score
            document: Document dictionary
            query_name: Name of the query being processed
            
        Returns:
            Boosted score
        """
        boost = 1.0
        
        # Boost for documents with potential reports
        potential_reports = document.get('potential_reports', [])
        if potential_reports:
            boost += 0.5 * len(potential_reports)
        
        # Boost for investor relations pages
        title = document.get('title', '').lower()
        if any(term in title for term in ['investor', 'financial', 'annual']):
            boost += 0.3
        
        # Year-specific boost
        if '2020' in query_name or '2021' in query_name or '2022' in query_name or '2023' in query_name or '2024' in query_name:
            year = re.search(r'(202[0-4])', query_name)
            if year:
                year_str = year.group(1)
                doc_text = ' '.join(document['tokens'])
                if year_str in doc_text:
                    boost += 0.4
        
        # PDF link boost
        for report in potential_reports:
            report_url = report.get('url', '').lower()
            if '.pdf' in report_url:
                boost += 0.2
        
        return score * boost
    
    def search_reports(self, crawl_data: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """
        Search for annual reports using BM25 ranking.
        
        Args:
            crawl_data: Crawled website data
            
        Returns:
            Dictionary of search results by query type
        """
        print("Building document corpus...")
        self._build_corpus(crawl_data)
        
        if not self.documents:
            print("No documents found in corpus.")
            return {}
        
        print(f"Built corpus with {len(self.documents)} documents")
        
        results = {}
        
        for query_name, query_info in self.report_queries.items():
            print(f"Searching for: {query_name}")
            
            query_terms = query_info['terms']
            query_weight = query_info['weight']
            
            scored_docs = []
            
            for doc in self.documents:
                score = self._bm25_score(query_terms, doc)
                boosted_score = self._boost_score(score, doc, query_name)
                final_score = boosted_score * query_weight
                
                if final_score > 0:
                    result = {
                        'url': doc['url'],
                        'title': doc['title'],
                        'score': final_score,
                        'potential_reports': doc['potential_reports'],
                        'query': query_name
                    }
                    scored_docs.append(result)
            
            # Sort by score and take top results
            scored_docs.sort(key=lambda x: x['score'], reverse=True)
            results[query_name] = scored_docs[:10]  # Top 10 results
        
        return results
    
    def extract_best_report_links(self, search_results: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Extract the best annual report links from search results.
        
        Args:
            search_results: BM25 search results
            
        Returns:
            List of best report links with metadata
        """
        all_reports = []
        seen_urls = set()
        
        for query_name, results in search_results.items():
            for result in results:
                potential_reports = result.get('potential_reports', [])
                
                for report in potential_reports:
                    report_url = report.get('url', '')
                    if report_url in seen_urls:
                        continue
                    
                    seen_urls.add(report_url)
                    
                    # Extract year from query or URL
                    year = None
                    year_match = re.search(r'(202[0-4])', query_name)
                    if year_match:
                        year = year_match.group(1)
                    else:
                        year_match = re.search(r'(202[0-4])', report_url)
                        if year_match:
                            year = year_match.group(1)
                    
                    report_info = {
                        'url': report_url,
                        'source_page': result['url'],
                        'source_title': result['title'],
                        'link_text': report.get('text', ''),
                        'link_title': report.get('title', ''),
                        'bm25_score': result['score'],
                        'relevance_score': report.get('relevance_score', 0),
                        'year': year,
                        'query_type': query_name,
                        'combined_score': result['score'] + report.get('relevance_score', 0)
                    }
                    
                    all_reports.append(report_info)
        
        # Sort by combined score
        all_reports.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return all_reports


class ReportAnalyzer:
    """
    Analyzer for post-processing and validating found reports.
    """
    
    @staticmethod
    def filter_by_year(reports: List[Dict], target_years: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Filter reports by target years.
        
        Args:
            reports: List of report dictionaries
            target_years: List of target years (default: 2020-2024)
            
        Returns:
            Dictionary of reports grouped by year
        """
        if target_years is None:
            target_years = ['2020', '2021', '2022', '2023', '2024']
        
        filtered_reports = defaultdict(list)
        
        for report in reports:
            year = report.get('year')
            if year in target_years:
                filtered_reports[year].append(report)
        
        return dict(filtered_reports)
    
    @staticmethod
    def deduplicate_reports(reports: List[Dict]) -> List[Dict]:
        """
        Remove duplicate reports based on URL similarity.
        
        Args:
            reports: List of report dictionaries
            
        Returns:
            Deduplicated list of reports
        """
        seen_urls = set()
        deduplicated = []
        
        for report in reports:
            url = report.get('url', '')
            
            # Simple deduplication by exact URL match
            if url not in seen_urls:
                seen_urls.add(url)
                deduplicated.append(report)
        
        return deduplicated
    
    @staticmethod
    def rank_reports_by_confidence(reports: List[Dict]) -> List[Dict]:
        """
        Rank reports by confidence score.
        
        Args:
            reports: List of report dictionaries
            
        Returns:
            Ranked list of reports
        """
        def confidence_score(report):
            score = report.get('combined_score', 0)
            
            # Boost PDF files
            if '.pdf' in report.get('url', '').lower():
                score += 1.0
            
            # Boost if link text contains 'annual report'
            link_text = report.get('link_text', '').lower()
            if 'annual report' in link_text:
                score += 0.5
            
            # Boost recent years
            year = report.get('year')
            if year in ['2023', '2024']:
                score += 0.3
            elif year in ['2021', '2022']:
                score += 0.1
            
            return score
        
        reports_with_confidence = []
        for report in reports:
            report_copy = report.copy()
            report_copy['confidence_score'] = confidence_score(report)
            reports_with_confidence.append(report_copy)
        
        reports_with_confidence.sort(key=lambda x: x['confidence_score'], reverse=True)
        return reports_with_confidence


# Example usage
def main():
    """Test the BM25 report finder."""
    # Load sample crawl data
    sample_data = {
        "https://example.com/investor": {
            "status": "crawled",
            "title": "Investor Relations - Annual Reports 2023",
            "description": "Find our latest annual reports and financial statements",
            "text_content": "Our annual report for 2023 provides comprehensive financial information...",
            "potential_reports": [
                {
                    "url": "https://example.com/annual-report-2023.pdf",
                    "text": "Annual Report 2023",
                    "relevance_score": 0.9
                }
            ]
        }
    }
    
    finder = BM25ReportFinder()
    results = finder.search_reports(sample_data)
    
    print("Search Results:")
    for query, docs in results.items():
        print(f"\n{query}:")
        for doc in docs[:3]:
            print(f"  {doc['title']} (Score: {doc['score']:.2f})")
    
    # Extract best links
    best_links = finder.extract_best_report_links(results)
    print(f"\nFound {len(best_links)} potential report links")


if __name__ == "__main__":
    main()
