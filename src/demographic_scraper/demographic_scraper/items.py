import scrapy

class DemographicScraperItem(scrapy.Item):
    link = scrapy.Field()
    male_ratio = scrapy.Field()
    female_ratio = scrapy.Field()
