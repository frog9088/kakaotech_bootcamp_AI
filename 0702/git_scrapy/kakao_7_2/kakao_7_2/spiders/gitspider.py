import scrapy


class GitspiderSpider(scrapy.Spider):
    name = "gitspider"
    allowed_domains = ["www.worldometers.info"]
    start_urls = ["https://www.worldometers.info/world-population/population-by-country/"]

    def parse(self, response):
        name = response.xpath('/html/body/div[2]/div[2]/div/div/div[1]/h1/text()').get()
        ID = response.xpath('//tr/td[1]/text()').getall()
        country = response.xpath('//tr/td[2]/a/text()').getall()


        for i in range(len(ID)):
            yield {
                'name': name,
                'ID': ID[i],     # Strip whitespace
                'country': country[i]  # Strip whitespace
            }