import pandas as pd
import scrapy
from scrapy.crawler import CrawlerProcess

def main():

    spider = QuotesSpider()

    process = CrawlerProcess()

    process.crawl(QuotesSpider)

    process.start()

    df = pd.DataFrame(data=spider.get_data())

    # 5

    with open("python-w-4-d-1-hw-result.csv", "wb") as f: df.to_csv(f)


class QuotesSpider(scrapy.Spider):

    _data = []

    name = "quotes"

    def start_requests(self):

        yield scrapy.Request(url="https://quotes.toscrape.com/", callback=self.parse)


    def parse(self, response):

        with open("quotes.html", "wb") as f: f.write(response.body)

        QuotesSpider._data = {

               # 1

               "quote": response.css("span.text::text").getall(),

               # 2

               "author": response.css("small.author::text").getall(),

               # 3

               "link": response.css("a[href*=author]::attr(href)").getall(),

               # 4

               "tags": response.css("meta.keywords::attr(content)").getall()

              }
    

    def get_data(self): return QuotesSpider._data


if __name__ == "__main__": main()


