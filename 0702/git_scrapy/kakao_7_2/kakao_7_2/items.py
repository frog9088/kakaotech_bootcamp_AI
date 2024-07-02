# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class Kakao72Item(scrapy.Item):
    # define the fields for your item here like:
    country = scrapy.Field()
    name = scrapy.Field()
    ID = scrapy.Field()

