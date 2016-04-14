import os.path
import json
import urlparse

ACCEPTED_FILETYPES = [
        'json',
        # 'csv'
]

class HistoryParser():
    def __init__(self, path):
        if not os.path.isfile(path):
            raise Exception("File not found.")
        if path.split(".")[-1] not in ACCEPTED_FILETYPES:
            raise Exception("Filetype not accepted.")

        self.path = path

    def _parse(self):
        with open(self.path) as data_file:
            data = json.load(data_file)
        return data

    def countVisitedPages(self):
        data = self._parse()
        visited = {}
        for entry in data:
            url = urlparse.urlparse(entry["url"]).netloc
            try:
                visited[url] = visited[url] + entry["visitCount"]
            except:
                visited[url] = entry["visitCount"]
        return visited

# hp = HistoryParser("../../example/andrei_history.json")
# hp.countVisitedPages()
