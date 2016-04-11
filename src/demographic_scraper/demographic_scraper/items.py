import scrapy

class DemographicScraperItem(scrapy.Item):
    link = scrapy.Field()
    male_ratio_alexa = scrapy.Field()
    female_ratio_alexa = scrapy.Field()
    male_ratio_quantcast = scrapy.Field()
    female_ratio_quantcast = scrapy.Field()
