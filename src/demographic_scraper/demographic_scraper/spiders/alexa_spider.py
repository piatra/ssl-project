import scrapy
from scrapy import Spider
import re
from urlparse import urlparse
from scrapy.selector import Selector
from demographic_scraper.items import DemographicScraperItem

class AlexaSpider(Spider):
    name = "alexa"

    def __init__(self, url="http://www.alexa.com/siteinfo/facebook.com", **kw):
        super(Spider, self).__init__(**kw)
        self.url = url
        self.allowed_domains = [re.sub(r'^www\.', '', urlparse(url).hostname)]

    def start_requests(self):
        return [scrapy.Request(self.url, callback=self.parse, dont_filter=False)]

    def parse(self, response):
        selector = Selector(response)
        bars = selector.css("#demographics-content .demo-col1 .pybar-bg")
        values = []
        for bar in bars:
            value = bar.css("span::attr(style)").extract()[0]
            value = int(re.search(r'\d+', value).group())
            print value
            values.append(value)
        male_ratio = float(values[0] + values[1]) / sum(values)
        female_ratio = float(values[2] + values[3]) / sum(values)
        return DemographicScraperItem(link=self.url,
                                      male_ratio=male_ratio,
                                      female_ratio=female_ratio)
