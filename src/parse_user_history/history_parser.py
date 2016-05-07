import os.path
import json

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
        self.unique = None

    def _parse(self):
        with open(self.path) as data_file:
            data = json.load(data_file)
        return data

    def get_frequency(self, url):
        if self.unique is None:
            self.unique = self.unique_links()

        return float(self.unique[url]) / sum(self.unique.values())


    def unique_links(self):
        data = self._parse()
        visited = {}
        for entry in data:
            url = entry["url"]
            if len(url.split(".")) > 2: # some links are actually browser addons addresses
                try:
                    visited[url] = visited[url] + entry["visitCount"]
                except:
                    visited[url] = entry["visitCount"]
        return visited
