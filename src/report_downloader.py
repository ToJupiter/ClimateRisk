"""
Report Downloader Module

This module handles downloading annual reports (PDF, HTML) and organizing them
in the output directory structure.
"""

import asyncio
import aiohttp
import aiofiles
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse, unquote
import hashlib
import mimetypes
from datetime import datetime
import json


class ReportDownloader:
    """
    Downloads annual reports and organizes them in a structured directory.
    
    This class handles downloading PDF files, HTML pages, and other document
    formats while maintaining proper file organization and metadata.
    """
    
    def __init__(self, output_dir: str = "output_data", 
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 timeout: int = 60):
        """
        Initialize the report downloader.
        
        Args:
            output_dir: Base output directory
            max_file_size: Maximum file size to download (bytes)
            timeout: Request timeout in seconds
        """
        self.output_dir = Path(output_dir)
        self.max_file_size = max_file_size
        self.timeout = timeout
        self.user_agent = 'FinancialReportDownloader/1.0'
        
        # Create output directory structure
        self._setup_directories()
        
        # Download metadata
        self.download_metadata = {}
        self.failed_downloads = []
    
    def _setup_directories(self):
        """Create the required directory structure."""
        # Main directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Year-based subdirectories
        years = ['2020', '2021', '2022', '2023', '2024']
        for year in years:
            (self.output_dir / "reports" / year).mkdir(exist_ok=True)
        
        print(f"Created directory structure in: {self.output_dir}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace problematic characters
        filename = filename.replace('/', '_').replace('\\', '_')
        filename = filename.replace(':', '_').replace('*', '_')
        filename = filename.replace('?', '_').replace('"', '_')
        filename = filename.replace('<', '_').replace('>', '_')
        filename = filename.replace('|', '_')
        
        # Limit length
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:200-len(ext)] + ext
        
        return filename
    
    def _generate_filename(self, url: str, company_name: str, year: str, 
                          content_type: str = None) -> str:
        """
        Generate a meaningful filename for the downloaded report.
        
        Args:
            url: Source URL
            company_name: Company name
            year: Report year
            content_type: MIME content type
            
        Returns:
            Generated filename
        """
        # Parse URL to get original filename
        parsed_url = urlparse(url)
        original_filename = os.path.basename(unquote(parsed_url.path))
        
        # Clean company name
        clean_company = self._sanitize_filename(company_name)
        clean_company = clean_company.replace(' ', '_')[:50]  # Limit length
        
        # Determine file extension
        if original_filename and '.' in original_filename:
            _, ext = os.path.splitext(original_filename)
        elif content_type:
            ext = mimetypes.guess_extension(content_type) or '.unknown'
        else:
            ext = '.pdf'  # Default assumption
        
        # Generate filename
        if original_filename and len(original_filename) < 100:
            # Use original filename if reasonable
            base_name = os.path.splitext(original_filename)[0]
            filename = f"{clean_company}_{year}_{base_name}{ext}"
        else:
            # Generate descriptive name
            filename = f"{clean_company}_annual_report_{year}{ext}"
        
        return self._sanitize_filename(filename)
    
    def _get_file_path(self, company_name: str, year: str, filename: str) -> Path:
        """
        Get the full file path for saving a report.
        
        Args:
            company_name: Company name
            year: Report year
            filename: Generated filename
            
        Returns:
            Full file path
        """
        # Organize by year, then by company
        year_dir = self.output_dir / "reports" / year
        company_dir = year_dir / self._sanitize_filename(company_name.replace(' ', '_'))
        company_dir.mkdir(exist_ok=True)
        
        return company_dir / filename
    
    async def download_report(self, url: str, company_name: str, year: str,
                            link_text: str = "", session: aiohttp.ClientSession = None) -> Dict:
        """
        Download a single report.
        
        Args:
            url: URL to download
            company_name: Company name
            year: Report year
            link_text: Original link text
            session: Optional aiohttp session
            
        Returns:
            Download result metadata
        """
        start_time = datetime.now()
        
        # Create session if not provided
        session_created = False
        if session is None:
            session = aiohttp.ClientSession()
            session_created = True
        
        try:
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            print(f"Downloading: {url}")
            
            async with session.get(url, headers=headers, timeout=self.timeout) as response:
                if response.status != 200:
                    return self._create_failure_result(url, company_name, year, 
                                                     f"HTTP {response.status}", start_time)
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_file_size:
                    return self._create_failure_result(url, company_name, year, 
                                                     "File too large", start_time)
                
                # Get content type
                content_type = response.headers.get('content-type', '').lower()
                
                # Generate filename and path
                filename = self._generate_filename(url, company_name, year, content_type)
                file_path = self._get_file_path(company_name, year, filename)
                
                # Download file
                downloaded_size = 0
                async with aiofiles.open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        downloaded_size += len(chunk)
                        
                        # Check size limit during download
                        if downloaded_size > self.max_file_size:
                            await f.close()
                            file_path.unlink()  # Delete partial file
                            return self._create_failure_result(url, company_name, year, 
                                                             "File too large during download", start_time)
                        
                        await f.write(chunk)
                
                # Create success result
                result = {
                    'status': 'success',
                    'url': url,
                    'company_name': company_name,
                    'year': year,
                    'link_text': link_text,
                    'filename': filename,
                    'file_path': str(file_path),
                    'file_size': downloaded_size,
                    'content_type': content_type,
                    'download_time': (datetime.now() - start_time).total_seconds(),
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"Successfully downloaded: {filename} ({downloaded_size:,} bytes)")
                return result
                
        except asyncio.TimeoutError:
            return self._create_failure_result(url, company_name, year, 
                                             "Download timeout", start_time)
        except Exception as e:
            return self._create_failure_result(url, company_name, year, 
                                             f"Error: {str(e)}", start_time)
        finally:
            if session_created:
                await session.close()
    
    def _create_failure_result(self, url: str, company_name: str, year: str, 
                              error: str, start_time: datetime) -> Dict:
        """
        Create a failure result dictionary.
        
        Args:
            url: Failed URL
            company_name: Company name
            year: Report year
            error: Error description
            start_time: Download start time
            
        Returns:
            Failure result metadata
        """
        result = {
            'status': 'failed',
            'url': url,
            'company_name': company_name,
            'year': year,
            'error': error,
            'download_time': (datetime.now() - start_time).total_seconds(),
            'timestamp': datetime.now().isoformat()
        }
        
        self.failed_downloads.append(result)
        print(f"Failed to download {url}: {error}")
        return result
    
    async def download_multiple_reports(self, reports: List[Dict], 
                                      max_concurrent: int = 5) -> List[Dict]:
        """
        Download multiple reports concurrently.
        
        Args:
            reports: List of report dictionaries with URL, company_name, year
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            List of download results
        """
        if not reports:
            return []
        
        print(f"Starting download of {len(reports)} reports...")
        
        # Create semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(report):
            async with semaphore:
                async with aiohttp.ClientSession() as session:
                    return await self.download_report(
                        url=report.get('url', ''),
                        company_name=report.get('company_name', ''),
                        year=report.get('year', ''),
                        link_text=report.get('link_text', ''),
                        session=session
                    )
        
        # Create download tasks
        tasks = [download_with_semaphore(report) for report in reports]
        
        # Execute downloads
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_result = self._create_failure_result(
                    reports[i].get('url', ''),
                    reports[i].get('company_name', ''),
                    reports[i].get('year', ''),
                    f"Task error: {str(result)}",
                    datetime.now()
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        # Save metadata
        await self.save_download_metadata(processed_results)
        
        print(f"Download completed. {sum(1 for r in processed_results if r['status'] == 'success')} successful, "
              f"{sum(1 for r in processed_results if r['status'] == 'failed')} failed")
        
        return processed_results
    
    async def save_download_metadata(self, results: List[Dict]):
        """
        Save download metadata to files.
        
        Args:
            results: List of download results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all results
        metadata_file = self.output_dir / "metadata" / f"download_results_{timestamp}.json"
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(results, indent=2))
        
        # Save successful downloads summary
        successful_downloads = [r for r in results if r['status'] == 'success']
        if successful_downloads:
            success_file = self.output_dir / "metadata" / f"successful_downloads_{timestamp}.json"
            async with aiofiles.open(success_file, 'w') as f:
                await f.write(json.dumps(successful_downloads, indent=2))
        
        # Save failed downloads
        failed_downloads = [r for r in results if r['status'] == 'failed']
        if failed_downloads:
            failed_file = self.output_dir / "metadata" / f"failed_downloads_{timestamp}.json"
            async with aiofiles.open(failed_file, 'w') as f:
                await f.write(json.dumps(failed_downloads, indent=2))
        
        print(f"Metadata saved to: {metadata_file}")
    
    def get_download_summary(self) -> Dict:
        """
        Get a summary of download statistics.
        
        Returns:
            Dictionary with download statistics
        """
        # Scan the reports directory to count files
        total_files = 0
        files_by_year = {}
        
        reports_dir = self.output_dir / "reports"
        if reports_dir.exists():
            for year_dir in reports_dir.iterdir():
                if year_dir.is_dir() and year_dir.name.isdigit():
                    year = year_dir.name
                    year_count = 0
                    
                    for company_dir in year_dir.iterdir():
                        if company_dir.is_dir():
                            company_files = list(company_dir.glob('*'))
                            year_count += len(company_files)
                    
                    files_by_year[year] = year_count
                    total_files += year_count
        
        return {
            'total_files': total_files,
            'files_by_year': files_by_year,
            'failed_downloads': len(self.failed_downloads),
            'output_directory': str(self.output_dir)
        }


# Helper function to convert selected reports to download format
def prepare_reports_for_download(selected_reports: Dict[str, List[Dict]], 
                                company_name: str) -> List[Dict]:
    """
    Convert selected reports to format expected by downloader.
    
    Args:
        selected_reports: Reports grouped by year from OpenAI selector
        company_name: Company name
        
    Returns:
        List of reports formatted for download
    """
    download_list = []
    
    for year, reports in selected_reports.items():
        for report in reports:
            download_item = {
                'url': report.get('url', ''),
                'company_name': company_name,
                'year': year,
                'link_text': report.get('link_text', ''),
                'ai_confidence': report.get('ai_confidence', 0),
                'final_score': report.get('final_score', 0)
            }
            download_list.append(download_item)
    
    return download_list


# Example usage
async def main():
    """Test the report downloader."""
    downloader = ReportDownloader()
    
    # Example reports to download
    test_reports = [
        {
            'url': 'https://example.com/annual-report-2023.pdf',
            'company_name': 'Test Company',
            'year': '2023',
            'link_text': 'Annual Report 2023'
        }
    ]
    
    # Download reports
    results = await downloader.download_multiple_reports(test_reports)
    
    # Print summary
    summary = downloader.get_download_summary()
    print("\nDownload Summary:")
    print(f"Total files: {summary['total_files']}")
    print(f"Files by year: {summary['files_by_year']}")
    print(f"Failed downloads: {summary['failed_downloads']}")


if __name__ == "__main__":
    asyncio.run(main())
