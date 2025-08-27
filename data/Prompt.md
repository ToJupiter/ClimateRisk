Here's a detailed plan for finding and downloading a company's annual report PDF, including steps, tool suggestions, and an AI agent prompt:

## Project Plan: Annual Report Retrieval

**Goal:** Find and download the annual report PDF of a specified company for the years 2020-2024.

---

### Step-by-Step Instructions:

**Step 1: Company Webpage Discovery**
*   **Action:** Programmatically search the web to identify the official website of the target company.
*   **Tool Suggestion:** OpenAI Search
*   **Output:** The primary URL of the company's official website.

**Step 2: Asynchronous Web Crawling and Annual Report Link Extraction**
*   **Action:** Implement an asynchronous web crawler to systematically explore all accessible branches (subpages and linked documents) of the company's identified website. During the crawl, an AI model will analyze the content of each page and its links to identify potential annual report documents or pages containing such links.
*   **Tool Suggestions:**
    *   **Asynchronous Crawler:** `Aiohttp` with `BeautifulSoup` (for more control and simpler projects). The key is to ensure an asynchronous approach to handle network requests efficiently.
    *   **AI for Link Identification:** `OpenAI API` (e.g., GPT-3.5 or GPT-4) for natural language understanding and pattern recognition. The AI will be prompted to identify links or text snippets that strongly indicate an annual report for the years 2020-2024 (e.g., "Annual Report 2022," "Financials 2021," "Investor Relations").
*   **Critical Fix for AsyncCrawler:**
    *   **Problem:** Asynchronous crawlers can sometimes retain stale page content in memory or cache if not explicitly reset or managed.
    *   **Solution:** Implement a clear state management system. Before initiating a crawl for a new URL, ensure that any previous page content, parsed data, or session-specific variables are explicitly cleared or re-initialized. For `aiohttp`, this might involve closing and reopening client sessions or explicitly clearing response content. For `Scrapy`, this is often handled by its architecture, but custom middleware might be needed for specific caching scenarios. A robust approach involves processing a page's data immediately after retrieval and then discarding the raw page content, rather than holding onto it.
*   **Output:** A list of potential URLs that directly link to annual reports or investor relations pages containing annual report links for the years 2020-2024.

**Step 3: Annual Report Download**
*   **Action:** For each identified URL from Step 2, attempt to download the content. Prioritize PDF files. If a direct PDF link isn't found, download the HTML page and analyze its content for embedded PDF links or other download mechanisms.
*   **Tool Suggestions:**
    *   **Downloading:** `Requests` library (for synchronous downloads) or `Aiohttp` (for asynchronous downloads, consistent with the crawler).
    *   **File Type Identification:** Check the `Content-Type` header of the HTTP response or the file extension in the URL.
    *   **PDF Extraction from HTML (if necessary):** If an HTML page contains an embedded link to a PDF, use `BeautifulSoup` again to parse the HTML and extract the PDF URL.
*   **Output:** Downloaded annual report files (preferably PDFs) for the years 2020-2024, stored in a designated directory.

---

### Suggestions for Tools:

*   **Asynchronous Web Crawling:** `Aiohttp` + `BeautifulSoup` (for more control over individual requests and parsing).
*   **HTML Parsing:** `BeautifulSoup4` (essential for extracting data from HTML).
*   **HTTP Requests:** `Requests` (for general HTTP interactions) or `Aiohttp` (for asynchronous HTTP interactions).
*   **AI Integration:** `openai` Python client library.
*   **File System Operations:** Python's built-in `os` and `shutil` modules.
*   **Logging:** Python's built-in `logging` module for tracking progress and errors.

---

### AI Agent Prompt (for Step 2 - Link Identification):

```
You are an intelligent web crawling assistant specialized in identifying financial documents, specifically annual reports.

Your task is to analyze the provided web page content and a list of extracted links. Identify any links or textual content that strongly indicates an annual report for a given company, specifically focusing on the years 2020, 2021, 2022, 2023, and 2024.

**Input:**
1.  `page_content`: The full HTML content of the current web page.
2.  `extracted_links`: A list of dictionaries, where each dictionary contains:
    *   `'text'`: The anchor text of the link (if available).
    *   `'href'`: The URL that the link points to.
3.  `company_name`: The name of the company being crawled.

**Output:**
Return a JSON array of dictionaries. Each dictionary should represent a potential annual report link and contain the following keys:
*   `'url'`: The full URL of the potential annual report document or page.
*   `'year'`: The identified year of the annual report (e.g., 2020, 2021, 2022, 2023, 2024). If a specific year cannot be confidently identified but the link is highly relevant, use "unknown".
*   `'confidence'`: A rating from 0.0 to 1.0 indicating how confident you are that this link leads to an annual report for the specified years.
*   `'reason'`: A brief explanation of why you identified this as a potential annual report link (e.g., "link text 'Annual Report 2022'", "URL contains 'financial-reports/2023'").

**Criteria for Identification:**
*   Look for keywords in link text, surrounding text, and URLs such as: "Annual Report", "Financial Report", "Investor Relations", "AR", "Form 10-K", "Geschäftsbericht" (if applicable), followed by years 2020, 2021, 2022, 2023, 2024.
*   Prioritize links that directly point to PDF files (e.g., URLs ending in `.pdf`).
*   Consider links within "Investor Relations" or "Financials" sections of the website.
*   Be mindful of different naming conventions companies might use.

**Example Input (for illustration - you will provide actual data):**
```json
{
    "page_content": "<html><body>...<a href='/investor-relations/annual-report-2022.pdf'>Annual Report 2022</a>...<a href='/financials'>Financials</a>...</body></html>",
    "extracted_links": [
        {"text": "Annual Report 2022", "href": "/investor-relations/annual-report-2022.pdf"},
        {"text": "Financials", "href": "/financials"},
        {"text": "Press Releases", "href": "/news/press"}
    ],
    "company_name": "Example Corp"
}
```

**Example Output (expected format):**
```json
[
    {
        "url": "https://www.examplecorp.com/investor-relations/annual-report-2022.pdf",
        "year": "2022",
        "confidence": 0.98,
        "reason": "Link text 'Annual Report 2022' and direct PDF URL."
    }
]
```
```