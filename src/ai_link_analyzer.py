"""
AI Link Analyzer Module

This module implements the AI agent specification from the project plan for 
identifying annual report links from crawled web pages. It integrates with 
OpenAI's API to analyze page content and extract relevant financial documents.

Implements the exact AI prompt format specified in the project plan.
"""

import json
import asyncio
from typing import List, Dict, Optional
from datetime import datetime
import openai
from openai import AsyncOpenAI


class AILinkAnalyzer:
    """
    AI-powered analyzer for identifying annual report links from web page content.
    
    This class implements the exact AI prompt specification from the project plan
    to analyze web pages and identify potential annual reports for years 2020-2024.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the AI link analyzer.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (default: gpt-3.5-turbo)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        
        # System prompt implementing the exact specification from project plan
        self.system_prompt = """You are an intelligent web crawling assistant specialized in identifying financial documents, specifically annual reports.

Your task is to analyze the provided web page content and a list of extracted links. Identify any links or textual content that strongly indicates an annual report for a given company, specifically focusing on the years 2020, 2021, 2022, 2023, and 2024.

**Criteria for Identification:**
- Look for keywords in link text, surrounding text, and URLs such as: "Annual Report", "Financial Report", "Investor Relations", "AR", "Form 10-K", "Geschäftsbericht" (if applicable), followed by years 2020, 2021, 2022, 2023, 2024.
- Prioritize links that directly point to PDF files (e.g., URLs ending in `.pdf`).
- Consider links within "Investor Relations" or "Financials" sections of the website.
- Be mindful of different naming conventions companies might use.

**Output Format:**
Return a JSON array of dictionaries. Each dictionary should represent a potential annual report link and contain the following keys:
- 'url': The full URL of the potential annual report document or page.
- 'year': The identified year of the annual report (e.g., 2020, 2021, 2022, 2023, 2024). If a specific year cannot be confidently identified but the link is highly relevant, use "unknown".
- 'confidence': A rating from 0.0 to 1.0 indicating how confident you are that this link leads to an annual report for the specified years.
- 'reason': A brief explanation of why you identified this as a potential annual report link.

Only return the JSON array, no additional text."""
    
    def _create_analysis_prompt(self, page_content: str, extracted_links: List[Dict], 
                               company_name: str) -> str:
        """
        Create the analysis prompt with page data.
        
        Args:
            page_content: HTML content of the page
            extracted_links: List of link dictionaries
            company_name: Company name
            
        Returns:
            Formatted prompt string
        """
        # Prepare the input data in exact format specified in project plan
        input_data = {
            "page_content": page_content[:10000],  # Limit content size for API
            "extracted_links": extracted_links,
            "company_name": company_name
        }
        
        prompt = f"""Analyze the following web page data to identify potential annual reports for {company_name}:

**Input:**
{json.dumps(input_data, indent=2, ensure_ascii=False)}

**Instructions:**
Focus on identifying links that lead to annual reports for years 2020, 2021, 2022, 2023, or 2024. Be particularly attentive to:
1. Direct PDF links with "annual report" or similar terms
2. Investor relations sections
3. Financial reporting pages

Remember to return only a JSON array with the exact format specified."""
        
        return prompt
    
    async def analyze_page(self, page_content: str, extracted_links: List[Dict], 
                          company_name: str) -> List[Dict]:
        """
        Analyze a single page for annual report links using AI.
        
        Args:
            page_content: HTML content of the page
            extracted_links: List of link dictionaries with 'text' and 'href' keys
            company_name: Name of the company
            
        Returns:
            List of identified annual report links with metadata
        """
        try:
            prompt = self._create_analysis_prompt(page_content, extracted_links, company_name)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=2000
            )
            
            # Extract and parse the JSON response
            content = response.choices[0].message.content.strip()
            
            # Handle potential formatting issues
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            # Parse JSON response
            try:
                reports = json.loads(content)
                if not isinstance(reports, list):
                    print(f"Warning: AI returned non-list response: {type(reports)}")
                    return []
                
                # Validate and enhance each report
                validated_reports = []
                for report in reports:
                    if self._validate_report_format(report):
                        # Add analysis metadata
                        report['ai_analyzed'] = True
                        report['analysis_timestamp'] = datetime.now().isoformat()
                        report['model_used'] = self.model
                        validated_reports.append(report)
                    else:
                        print(f"Warning: Invalid report format: {report}")
                
                return validated_reports
                
            except json.JSONDecodeError as e:
                print(f"Error parsing AI response JSON: {e}")
                print(f"AI Response: {content}")
                return []
                
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            return []
    
    def _validate_report_format(self, report: Dict) -> bool:
        """
        Validate that a report has the required format.
        
        Args:
            report: Report dictionary to validate
            
        Returns:
            True if format is valid
        """
        required_keys = ['url', 'year', 'confidence', 'reason']
        
        if not all(key in report for key in required_keys):
            return False
        
        # Validate confidence is numeric and in range
        try:
            confidence = float(report['confidence'])
            if not 0.0 <= confidence <= 1.0:
                return False
        except (ValueError, TypeError):
            return False
        
        # Validate URL is not empty
        if not report['url'].strip():
            return False
        
        return True
    
    async def analyze_multiple_pages(self, pages_data: List[Dict], 
                                   company_name: str) -> Dict[str, List[Dict]]:
        """
        Analyze multiple pages concurrently for annual report links.
        
        Args:
            pages_data: List of page data dictionaries with 'page_content', 
                       'extracted_links', and 'source_url'
            company_name: Name of the company
            
        Returns:
            Dictionary mapping source URLs to lists of identified reports
        """
        print(f"🧠 Starting AI analysis of {len(pages_data)} pages for {company_name}")
        
        # Create analysis tasks
        tasks = []
        for page_data in pages_data:
            task = self.analyze_page(
                page_data['page_content'],
                page_data['extracted_links'],
                company_name
            )
            tasks.append((page_data['source_url'], task))
        
        # Execute analysis concurrently
        results = {}
        completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
        
        for i, result in enumerate(completed_tasks):
            source_url = tasks[i][0]
            if isinstance(result, Exception):
                print(f"❌ Error analyzing {source_url}: {result}")
                results[source_url] = []
            else:
                results[source_url] = result
                print(f"✅ Analyzed {source_url}: found {len(result)} potential reports")
        
        return results
    
    def aggregate_results(self, analysis_results: Dict[str, List[Dict]]) -> List[Dict]:
        """
        Aggregate and deduplicate results from multiple pages.
        
        Args:
            analysis_results: Dictionary of analysis results by source URL
            
        Returns:
            Aggregated and deduplicated list of reports
        """
        all_reports = []
        seen_urls = set()
        
        for source_url, reports in analysis_results.items():
            for report in reports:
                report_url = report['url']
                
                # Deduplicate by URL
                if report_url not in seen_urls:
                    seen_urls.add(report_url)
                    # Add source information
                    report['source_page'] = source_url
                    all_reports.append(report)
                else:
                    # If we've seen this URL, check if this has higher confidence
                    existing_report = next((r for r in all_reports if r['url'] == report_url), None)
                    if existing_report and report['confidence'] > existing_report['confidence']:
                        # Replace with higher confidence version
                        all_reports.remove(existing_report)
                        report['source_page'] = source_url
                        all_reports.append(report)
        
        # Sort by confidence descending, then by year
        all_reports.sort(key=lambda x: (x['confidence'], x['year'] if x['year'] != 'unknown' else '0'), reverse=True)
        
        return all_reports
    
    def filter_by_year(self, reports: List[Dict], target_years: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Filter and group reports by year.
        
        Args:
            reports: List of report dictionaries
            target_years: List of target years (default: 2020-2024)
            
        Returns:
            Dictionary mapping years to lists of reports
        """
        if target_years is None:
            target_years = ['2020', '2021', '2022', '2023', '2024']
        
        filtered_reports = {}
        
        for year in target_years:
            year_reports = [r for r in reports if r['year'] == year]
            if year_reports:
                filtered_reports[year] = year_reports
        
        # Also include unknown year reports if they have high confidence
        unknown_reports = [r for r in reports if r['year'] == 'unknown' and r['confidence'] >= 0.7]
        if unknown_reports:
            filtered_reports['unknown'] = unknown_reports
        
        return filtered_reports
    
    def create_summary_report(self, reports: List[Dict], company_name: str) -> Dict:
        """
        Create a summary report of the AI analysis.
        
        Args:
            reports: List of identified reports
            company_name: Company name
            
        Returns:
            Summary dictionary
        """
        by_year = self.filter_by_year(reports)
        
        summary = {
            'company_name': company_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_reports_found': len(reports),
            'reports_by_year': {year: len(reports) for year, reports in by_year.items()},
            'highest_confidence_report': max(reports, key=lambda x: x['confidence']) if reports else None,
            'pdf_reports_count': len([r for r in reports if r['url'].lower().endswith('.pdf')]),
            'average_confidence': sum(r['confidence'] for r in reports) / len(reports) if reports else 0,
            'years_covered': list(by_year.keys())
        }
        
        return summary


# Example usage and testing
async def test_ai_analyzer():
    """Test the AI link analyzer with sample data."""
    # Note: You need to provide a real OpenAI API key
    api_key = "your-openai-api-key-here"
    
    if api_key == "your-openai-api-key-here":
        print("⚠️ Please provide a real OpenAI API key to test the AI analyzer")
        return
    
    analyzer = AILinkAnalyzer(api_key)
    
    # Sample page data (in the format expected by the analyzer)
    sample_page_data = {
        'page_content': '''
        <html>
        <head><title>Investor Relations - Apple Inc</title></head>
        <body>
        <h1>Investor Relations</h1>
        <div class="financial-reports">
            <h2>Annual Reports</h2>
            <ul>
                <li><a href="/investor/annual-report-2023.pdf">Annual Report 2023</a></li>
                <li><a href="/investor/annual-report-2022.pdf">Annual Report 2022</a></li>
                <li><a href="/investor/financial-statements-2021.pdf">Financial Statements 2021</a></li>
            </ul>
        </div>
        </body>
        </html>
        ''',
        'extracted_links': [
            {'text': 'Annual Report 2023', 'href': '/investor/annual-report-2023.pdf'},
            {'text': 'Annual Report 2022', 'href': '/investor/annual-report-2022.pdf'},
            {'text': 'Financial Statements 2021', 'href': '/investor/financial-statements-2021.pdf'},
            {'text': 'Contact Us', 'href': '/contact'}
        ],
        'source_url': 'https://investor.apple.com'
    }
    
    # Test single page analysis
    print("🧠 Testing AI analysis on sample page...")
    reports = await analyzer.analyze_page(
        sample_page_data['page_content'],
        sample_page_data['extracted_links'],
        "Apple Inc"
    )
    
    print(f"Found {len(reports)} potential reports:")
    for report in reports:
        print(f"- {report['url']} (Year: {report['year']}, Confidence: {report['confidence']:.2f})")
        print(f"  Reason: {report['reason']}")
    
    # Create summary
    summary = analyzer.create_summary_report(reports, "Apple Inc")
    print(f"\nSummary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    asyncio.run(test_ai_analyzer()) 