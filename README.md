# Annual Report Crawler

A comprehensive system to automatically find and download annual reports from company websites. This crawler uses advanced techniques including BM25 search algorithm and optional OpenAI integration for intelligent link selection.

## Features

- **🔍 Website Discovery**: Automatically finds company websites using search APIs
- **🕷️ Intelligent Crawling**: Asynchronous crawler that respects robots.txt and follows best practices
- **🎯 BM25 Search**: Advanced search algorithm specifically tuned for financial documents
- **🤖 AI-Powered Selection**: Optional OpenAI integration for intelligent link selection
- **📥 Smart Downloads**: Organized file management with metadata tracking
- **📊 Comprehensive Reporting**: Detailed progress tracking and result summaries

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd ClimateRisk

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Process all companies from the ESG CSV file (first 5 companies)
python main.py

# Process a specific company
python main.py --company "Apple Inc" --country "US"

# Process companies from a custom CSV file
python main.py --csv data/companies.csv

# Use OpenAI for better link selection
python main.py --openai-key "your-openai-api-key"
```

### 3. Configuration

Edit `config.json` to customize behavior:

```json
{
  "target_years": ["2020", "2021", "2022", "2023", "2024"],
  "crawler": {
    "max_pages": 50,
    "n_workers": 10
  },
  "openai": {
    "api_key": "your-openai-api-key",
    "model": "gpt-3.5-turbo"
  },
  "download": {
    "output_dir": "output_data",
    "max_concurrent": 3
  },
  "max_companies": 5
}
```

## System Architecture

### Components

1. **WebsiteFinder** (`src/website_finder.py`)
   - Finds company websites using OpenAI Web Search
   - Handles company name normalization and validation
   - Supports geographic location-aware search

2. **EnhancedAsyncCrawler** (`src/enhanced_crawler.py`)
   - Asynchronous web crawler with content analysis
   - Respects robots.txt and implements rate limiting
   - Extracts and scores potential report links

3. **BM25ReportFinder** (`src/report_finder.py`)
   - Implements BM25 search algorithm for document ranking
   - Specialized for financial document discovery
   - Includes post-processing and filtering

4. **OpenAIReportSelector** (`src/openai_selector.py`)
   - Uses OpenAI API for intelligent link selection
   - Provides reasoning for selections
   - Fallback to rule-based selection if API unavailable

5. **ReportDownloader** (`src/report_downloader.py`)
   - Downloads reports with proper file organization
   - Handles various document formats (PDF, HTML, etc.)
   - Tracks metadata and provides progress reporting

### Workflow

```
CSV/Input → Website Finding → Crawling → BM25 Search → AI Selection → Download
```

## Output Structure

```
output_data/
├── reports/
│   ├── 2020/
│   │   └── Company_Name/
│   │       └── Company_Name_annual_report_2020.pdf
│   ├── 2021/
│   └── ...
├── metadata/
│   ├── download_results_YYYYMMDD_HHMMSS.json
│   ├── successful_downloads_YYYYMMDD_HHMMSS.json
│   └── failed_downloads_YYYYMMDD_HHMMSS.json
└── logs/
    └── crawler_results_YYYYMMDD_HHMMSS.json
```

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --config CONFIG         Configuration file path (default: config.json)
  --csv CSV              CSV file with companies
  --company COMPANY      Single company name to process
  --country COUNTRY      Country code for single company
  --openai-key KEY       OpenAI API key for AI selection
  --max-companies N      Maximum number of companies to process
  --help                 Show help message
```

## Configuration Options

### Crawler Settings
- `max_pages`: Maximum pages to crawl per website
- `n_workers`: Number of concurrent crawler workers
- `user_agent`: User agent string for requests

### BM25 Settings
- `k1`: Term frequency saturation parameter (default: 1.2)
- `b`: Document length normalization (default: 0.75)

### OpenAI Settings
- `api_key`: Your OpenAI API key
- `model`: Model to use (gpt-3.5-turbo, gpt-4, etc.)

### Download Settings
- `output_dir`: Base directory for downloads
- `max_file_size`: Maximum file size to download (bytes)
- `max_concurrent`: Maximum concurrent downloads

## Input Data Format

### CSV Format
The system expects a CSV file with the following columns:
- `Company Name`: Name of the company
- `HQ`: Country code (optional, helps with search accuracy)

Example:
```csv
Company Name,HQ,Industry Group
Apple Inc,US,Technology
Microsoft Corporation,US,Technology
```

## Error Handling

The system includes comprehensive error handling:
- **Network errors**: Automatic retries with exponential backoff
- **Robots.txt compliance**: Respects website crawling policies
- **File size limits**: Prevents downloading excessively large files
- **API failures**: Graceful fallback to rule-based selection

## Performance Considerations

- **Asynchronous operations**: All I/O operations are non-blocking
- **Rate limiting**: Respects server load and robots.txt
- **Memory management**: Streams large files and limits content storage
- **Concurrent limits**: Configurable concurrency for downloads and crawling

## Troubleshooting

### Common Issues

1. **No websites found**
   - Check company name spelling
   - Try adding country code
   - Check internet connection

2. **No reports found**
   - Increase `max_pages` in crawler config
   - Check if website has investor relations section
   - Try different target years

3. **Download failures**
   - Check file size limits
   - Verify network connectivity
   - Check disk space

4. **OpenAI API errors**
   - Verify API key is valid
   - Check API usage limits
   - System will fallback to rule-based selection

### Debug Mode

Add debug logging by modifying the configuration:
```json
{
  "crawler": {
    "debug": true
  }
}
```

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `output_data/logs/`
3. Create an issue with detailed error information