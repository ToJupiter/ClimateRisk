"""
OpenAI-powered Report Link Selector

This module uses OpenAI's API to intelligently select the most relevant
annual report links from the candidates found by the crawler and BM25 algorithm.
"""

import asyncio
import json
from typing import List, Dict, Optional, Tuple
import openai
from dataclasses import dataclass
import re


@dataclass
class ReportCandidate:
    """Data class representing a candidate annual report link."""
    url: str
    link_text: str
    source_page: str
    source_title: str
    year: Optional[str]
    bm25_score: float
    relevance_score: float
    confidence_score: float


class OpenAIReportSelector:
    """
    Uses OpenAI API to intelligently select the best annual report links.
    
    This class leverages GPT models to understand context and make intelligent
    decisions about which links are most likely to be actual annual reports.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the OpenAI selector.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-3.5-turbo)
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
        
        # System prompt for annual report selection
        self.system_prompt = """You are an expert financial analyst tasked with identifying genuine annual report links from a list of candidates. 

Your job is to analyze each link and determine:
1. Whether it's likely to be an actual annual report (not just news, press releases, or other documents)
2. What year the report is for (2020-2024)
3. The confidence level (1-10) that this is a legitimate annual report

Consider these factors:
- Link text containing "annual report", "10-K", "Form 10-K", "financial statements"
- URLs containing "annual", "report", "investor", "sec", "financial"
- PDF files are more likely to be actual reports
- Links from investor relations pages are more credible
- Avoid: news articles, press releases, quarterly reports, sustainability reports (unless specifically annual)

Respond with a JSON array of objects, each containing:
{
  "url": "the URL",
  "is_annual_report": boolean,
  "year": "YYYY" or null,
  "confidence": 1-10,
  "reasoning": "brief explanation"
}"""
    
    async def analyze_candidates(self, candidates: List[ReportCandidate], 
                               company_name: str, 
                               target_years: List[str] = None) -> List[Dict]:
        """
        Analyze candidate links using OpenAI to select the best annual reports.
        
        Args:
            candidates: List of candidate report links
            company_name: Name of the company
            target_years: Target years for reports (default: 2020-2024)
            
        Returns:
            List of analyzed and filtered report links
        """
        if target_years is None:
            target_years = ['2020', '2021', '2022', '2023', '2024']
        
        if not candidates:
            return []
        
        # Limit to top candidates to avoid token limits
        top_candidates = candidates[:20]
        
        # Prepare the user prompt
        user_prompt = self._create_user_prompt(top_candidates, company_name, target_years)
        
        try:
            # Make OpenAI API call
            response = await self._call_openai_api(user_prompt)
            
            # Parse and validate response
            analyzed_links = self._parse_openai_response(response, top_candidates)
            
            # Filter and rank results
            filtered_links = self._filter_and_rank_results(analyzed_links, target_years)
            
            return filtered_links
            
        except Exception as e:
            print(f"Error in OpenAI analysis: {e}")
            # Fallback to rule-based selection
            return self._fallback_selection(top_candidates, target_years)
    
    def _create_user_prompt(self, candidates: List[ReportCandidate], 
                          company_name: str, target_years: List[str]) -> str:
        """
        Create the user prompt for OpenAI analysis.
        
        Args:
            candidates: Candidate links
            company_name: Company name
            target_years: Target years
            
        Returns:
            Formatted user prompt
        """
        prompt = f"""Analyze these potential annual report links for {company_name}.

Target years: {', '.join(target_years)}

Candidates:
"""
        
        for i, candidate in enumerate(candidates, 1):
            prompt += f"""
{i}. URL: {candidate.url}
   Link Text: "{candidate.link_text}"
   Source Page: {candidate.source_page}
   Source Title: "{candidate.source_title}"
   Detected Year: {candidate.year or 'Unknown'}
   BM25 Score: {candidate.bm25_score:.2f}
   Relevance Score: {candidate.relevance_score:.2f}
"""
        
        prompt += f"""
Please analyze each link and determine if it's a genuine annual report for {company_name} from {', '.join(target_years)}.
"""
        
        return prompt
    
    async def _call_openai_api(self, user_prompt: str) -> str:
        """
        Make the OpenAI API call.
        
        Args:
            user_prompt: The user prompt
            
        Returns:
            OpenAI response text
        """
        response = await asyncio.to_thread(
            openai.ChatCompletion.create,
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _parse_openai_response(self, response: str, candidates: List[ReportCandidate]) -> List[Dict]:
        """
        Parse the OpenAI response and match with original candidates.
        
        Args:
            response: OpenAI response text
            candidates: Original candidate list
            
        Returns:
            List of analyzed links
        """
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analyzed = json.loads(json_str)
            else:
                # Try to parse entire response as JSON
                analyzed = json.loads(response)
            
            # Match analyzed results with original candidates
            results = []
            candidate_dict = {c.url: c for c in candidates}
            
            for analysis in analyzed:
                url = analysis.get('url', '')
                if url in candidate_dict:
                    candidate = candidate_dict[url]
                    result = {
                        'candidate': candidate,
                        'is_annual_report': analysis.get('is_annual_report', False),
                        'ai_year': analysis.get('year'),
                        'ai_confidence': analysis.get('confidence', 0),
                        'ai_reasoning': analysis.get('reasoning', ''),
                        'url': url,
                        'link_text': candidate.link_text,
                        'year': analysis.get('year') or candidate.year,
                        'final_score': self._calculate_final_score(candidate, analysis)
                    }
                    results.append(result)
            
            return results
            
        except json.JSONDecodeError as e:
            print(f"Error parsing OpenAI response: {e}")
            print(f"Response: {response}")
            return []
    
    def _calculate_final_score(self, candidate: ReportCandidate, analysis: Dict) -> float:
        """
        Calculate final score combining AI analysis with existing scores.
        
        Args:
            candidate: Original candidate
            analysis: AI analysis result
            
        Returns:
            Final combined score
        """
        if not analysis.get('is_annual_report', False):
            return 0.0
        
        # Base score from AI confidence
        score = analysis.get('confidence', 0) / 10.0
        
        # Add existing scores
        score += candidate.bm25_score * 0.3
        score += candidate.relevance_score * 0.2
        score += candidate.confidence_score * 0.2
        
        # Boost for PDF files
        if '.pdf' in candidate.url.lower():
            score += 0.5
        
        # Boost for recent years
        year = analysis.get('year') or candidate.year
        if year in ['2023', '2024']:
            score += 0.3
        elif year in ['2021', '2022']:
            score += 0.1
        
        return score
    
    def _filter_and_rank_results(self, analyzed_links: List[Dict], 
                                target_years: List[str]) -> List[Dict]:
        """
        Filter and rank the analyzed results.
        
        Args:
            analyzed_links: Links analyzed by AI
            target_years: Target years for filtering
            
        Returns:
            Filtered and ranked results
        """
        # Filter for annual reports only
        annual_reports = [
            link for link in analyzed_links 
            if link.get('is_annual_report', False) and 
            link.get('ai_confidence', 0) >= 5  # Minimum confidence threshold
        ]
        
        # Filter by target years if year is detected
        if target_years:
            filtered_reports = []
            for report in annual_reports:
                year = report.get('year')
                if not year or year in target_years:
                    filtered_reports.append(report)
            annual_reports = filtered_reports
        
        # Sort by final score
        annual_reports.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        
        return annual_reports
    
    def _fallback_selection(self, candidates: List[ReportCandidate], 
                          target_years: List[str]) -> List[Dict]:
        """
        Fallback selection method when OpenAI API fails.
        
        Args:
            candidates: Candidate links
            target_years: Target years
            
        Returns:
            Selected links using rule-based approach
        """
        print("Using fallback rule-based selection...")
        
        results = []
        for candidate in candidates:
            # Simple rule-based scoring
            score = 0.0
            
            # Check for annual report keywords
            link_text = candidate.link_text.lower()
            if 'annual report' in link_text:
                score += 3.0
            elif 'annual' in link_text and 'report' in link_text:
                score += 2.0
            elif 'financial report' in link_text:
                score += 2.0
            
            # Check for year
            year_match = None
            for year in target_years:
                if year in link_text or year in candidate.url:
                    score += 1.0
                    year_match = year
                    break
            
            # Check for PDF
            if '.pdf' in candidate.url.lower():
                score += 1.0
            
            # Only include if minimum score reached
            if score >= 2.0:
                result = {
                    'candidate': candidate,
                    'is_annual_report': True,
                    'ai_year': year_match,
                    'ai_confidence': min(int(score), 10),
                    'ai_reasoning': 'Rule-based fallback selection',
                    'url': candidate.url,
                    'link_text': candidate.link_text,
                    'year': year_match or candidate.year,
                    'final_score': score + candidate.confidence_score
                }
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.get('final_score', 0), reverse=True)
        return results
    
    async def select_best_reports_by_year(self, candidates: List[ReportCandidate], 
                                        company_name: str, 
                                        target_years: List[str] = None,
                                        max_per_year: int = 3) -> Dict[str, List[Dict]]:
        """
        Select the best annual reports grouped by year.
        
        Args:
            candidates: Candidate links
            company_name: Company name
            target_years: Target years (default: 2020-2024)
            max_per_year: Maximum reports per year
            
        Returns:
            Dictionary of reports grouped by year
        """
        if target_years is None:
            target_years = ['2020', '2021', '2022', '2023', '2024']
        
        # Get analyzed results
        analyzed_links = await self.analyze_candidates(candidates, company_name, target_years)
        
        # Group by year
        reports_by_year = {}
        for year in target_years:
            year_reports = [
                report for report in analyzed_links 
                if report.get('year') == year
            ]
            reports_by_year[year] = year_reports[:max_per_year]
        
        return reports_by_year


# Example usage
async def main():
    """Test the OpenAI report selector."""
    # Example candidates
    candidates = [
        ReportCandidate(
            url="https://example.com/annual-report-2023.pdf",
            link_text="Annual Report 2023",
            source_page="https://example.com/investor",
            source_title="Investor Relations",
            year="2023",
            bm25_score=2.5,
            relevance_score=0.9,
            confidence_score=3.2
        ),
        ReportCandidate(
            url="https://example.com/news-2023.html",
            link_text="Company News 2023",
            source_page="https://example.com/news",
            source_title="Latest News",
            year="2023",
            bm25_score=1.1,
            relevance_score=0.3,
            confidence_score=1.5
        )
    ]
    
    # Note: Replace with actual OpenAI API key
    selector = OpenAIReportSelector("your-openai-api-key")
    
    try:
        results = await selector.analyze_candidates(candidates, "Example Company")
        
        print("Selected Annual Reports:")
        for result in results:
            if result['is_annual_report']:
                print(f"Year {result['year']}: {result['link_text']}")
                print(f"  URL: {result['url']}")
                print(f"  Confidence: {result['ai_confidence']}/10")
                print(f"  Reasoning: {result['ai_reasoning']}")
                print()
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
