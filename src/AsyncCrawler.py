import asyncio
import aiohttp
import nest_asyncio
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup

# Apply nest_asyncio to allow running asyncio in Jupyter Notebook
nest_asyncio.apply()

# --- Configuration ---
# Set a user agent to identify your crawler
USER_AGENT = 'MyAsyncCrawler/1.0'
# Cache for robot.txt parsers
robots_parsers = {}
# Dictionary to store the hyperlink tree
link_tree = {}
# Set of visited URLs to avoid re-crawling and infinite loops
visited_urls = set()

async def can_fetch(session, url, user_agent):
    """
    Checks if the url can be fetched according to the site's robots.txt.
    Caches the RobotFileParser for each domain.
    """
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

    # Get or create a parser for the domain
    if robots_url not in robots_parsers:
        parser = RobotFileParser()
        try:
            async with session.get(robots_url) as response:
                if response.status == 200:
                    text = await response.text()
                    parser.parse(text.splitlines())
                # If robots.txt is not found or gives an error, we assume we can crawl.
                else:
                    parser.allow_all = True
        except Exception:
             # On any exception, assume crawling is allowed.
            parser.allow_all = True
        robots_parsers[robots_url] = parser

    parser = robots_parsers[robots_url]
    return parser.can_fetch(user_agent, url)

async def worker(name, queue, session):
    """
    A worker that processes URLs from a queue.
    """
    while True:
        # Get a URL from the queue
        url = await queue.get()

        # Check if already visited
        if url in visited_urls:
            queue.task_done()
            continue

        visited_urls.add(url)

        # Check robots.txt before fetching
        if not await can_fetch(session, url, USER_AGENT):
            print(f"Worker {name}: Skipped (robots.txt): {url}")
            link_tree[url] = {"status": "skipped_robots"}
            queue.task_done()
            continue

        print(f"Worker {name}: Crawling {url}")

        try:
            # Set headers for the request
            headers = {'User-Agent': USER_AGENT}
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 200 and 'text/html' in response.headers.get('content-type', ''):
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    found_links = set()
                    base_url_parts = urlparse(url)

                    for a_tag in soup.find_all('a', href=True):
                        href = a_tag['href']
                        # Join relative URLs with the base URL
                        full_url = urljoin(url, href)

                        # Only follow links within the same domain
                        if urlparse(full_url).netloc == base_url_parts.netloc:
                            found_links.add(full_url)
                            # Add new, unvisited links to the queue
                            if full_url not in visited_urls:
                                await queue.put(full_url)

                    link_tree[url] = {"status": "crawled", "links": list(found_links)}
                else:
                    link_tree[url] = {"status": f"failed_status_{response.status}"}

        except Exception as e:
            print(f"Worker {name}: Error crawling {url}: {e}")
            link_tree[url] = {"status": "error", "reason": str(e)}
        finally:
            # Notify the queue that the task is done
            queue.task_done()

async def crawl(start_url, max_pages=50, n_workers=10):
    """
    Main function to orchestrate the crawling process.
    """
    queue = asyncio.Queue()
    queue.put_nowait(start_url)

    async with aiohttp.ClientSession() as session:
        # Create worker tasks
        tasks = []
        for i in range(n_workers):
            task = asyncio.create_task(worker(f'worker-{i+1}', queue, session))
            tasks.append(task)

        # Wait until the queue is fully processed, up to max_pages
        while len(visited_urls) < max_pages:
            try:
                await asyncio.wait_for(queue.join(), timeout=10.0)
                if queue.empty():
                    break
            except asyncio.TimeoutError:
                print("Timeout reached, some pages might still be in queue. Stopping.")
                break

        # Cancel all worker tasks
        for task in tasks:
            task.cancel()

        # Wait for all tasks to be cancelled
        await asyncio.gather(*tasks, return_exceptions=True)

        print(f"\nCrawling finished. Visited {len(visited_urls)} pages.")


# --- Main Execution ---
async def main():
    # Define the starting URL for the crawl
    START_URL = "https://www.lseg.com/en"
    # Define the maximum number of pages to crawl
    MAX_PAGES = 5000
    # Define the number of concurrent workers
    NUM_WORKERS = 50

    print(f"Starting crawl at: {START_URL}")
    print(f"Max pages: {MAX_PAGES}, Workers: {NUM_WORKERS}")

    await crawl(start_url=START_URL, max_pages=MAX_PAGES, n_workers=NUM_WORKERS)

    # Print the resulting link tree
    import json
    print("\n--- Hyperlink Tree ---")
    print(json.dumps(link_tree, indent=2))

# In a Jupyter notebook, you would run the main function like this:
if __name__ == "__main__":
    asyncio.run(main())
    print(len(link_tree))