import requests
from typing import Dict, List

from bs4 import BeautifulSoup


BASE_URL = 'https://lite.duckduckgo.com'


class SearchRequest:
    def __init__(self, query: str, timerange: str, region: str):
        self.query = query
        self.timerange = timerange
        self.region = region


class SearchResponse:
    def __init__(self, status: int, html: str, url: str):
        self.status = status
        self.html = html
        self.url = url


class SearchResult:
    def __init__(self, title: str, body: str, url: str):
        self.title = title
        self.body = body
        self.url = url


def get_html(search: SearchRequest) -> SearchResponse:
    query = search.query[:495]  # DDG limit
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'text/html,application/xhtml+xml,application/xmlq=0.9,image/avif,image/webp,image/apng,*/*q=0.8,application/signed-exchangev=b3q=0.7',
        'AcceptEncoding': 'gzip, deflate, br',
    }
    data = {
        'q': query,
        'df': search.timerange,
        'kl': search.region,
    }
    response = requests.post(f'{BASE_URL}/lite/', headers=headers, data=data)
    if not response.ok:
        raise Exception(f'Failed to fetch: {response.status_code} {response.reason}')
    return SearchResponse(response.status_code, response.text, response.url)


def html_to_search_results(html: str, num_results: int) -> List[SearchResult]:
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    zero_click_link = soup.select_one('table:nth-of-type(2) tr td a[rel="nofollow"]')
    if zero_click_link:
        title = zero_click_link.text
        body = soup.select_one('table:nth-of-type(2) tr:nth-of-type(2)').text.strip()
        url = zero_click_link['href']
        results.append(SearchResult(title, body, url))
    upper_bound = num_results - 1 if zero_click_link else num_results
    web_links = soup.select('table:nth-of-type(3) .result-link')[:upper_bound]
    web_snippets = soup.select('table:nth-of-type(3) .result-snippet')[:upper_bound]
    for link, snippet in zip(web_links, web_snippets):
        title = link.text
        body = snippet.text.strip()
        url = link['href']
        results.append(SearchResult(title, body, url))
    return results


def web_search(search: SearchRequest, num_results: int) -> List[SearchResult]:
    response = get_html(search)
    if response.url == f'{BASE_URL}/lite/':
        return html_to_search_results(response.html, num_results)
    else:
        # TODO: Implement web page text extraction
        return []