import json
import scrapy
from scrapy import Spider
import re
from urlparse import urlparse
from scrapy.selector import Selector
from demographic_scraper.demographic_scraper.items import DemographicScraperItem


class QuantcastSpider(Spider):
    name = "quantcast"

    def __init__(self, url="500px.com", **kw):
        super(Spider, self).__init__(**kw)
        self.request_url = url
        self.url = "https://www.quantcast.com/" + url
        self.allowed_domains = [re.sub(r'^www\.', '', urlparse(self.url).hostname)]

    def start_requests(self):
        return [scrapy.Request(self.url, callback=self.parse, dont_filter=False)]

    def parse(self, response):
        selector = Selector(response)
        demographics = selector.xpath('//script[@key="demographicsData"]/text()').extract()
        if not demographics:
            return
        # This is a valid JSON.
        demographics = json.loads(demographics[0].strip())

        male_ratio = female_ratio = 0
        for demographics_data in demographics.get('WEB', []):
            if demographics_data['readable_name'] != 'Gender':
                continue
            for bar in demographics_data.get('bars'):
                if bar['title'] == 'Male':
                    male_ratio = bar['composition']
                else:
                    female_ratio = bar['composition']
            # We got what we were looking for.
            break

        return DemographicScraperItem(link=self.request_url,
                                      male_ratio_quantcast=male_ratio,
                                      female_ratio_quantcast=female_ratio)
