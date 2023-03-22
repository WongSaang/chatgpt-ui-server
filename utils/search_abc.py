class SearchRequest:
    def __init__(self, query: str, timerange: str = None, region: str = None):
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
