import re

import numpy as np
import pandas as pd
import scrapy
from scrapy.crawler import CrawlerProcess


def main():

    spider = HouseSpider()

    process = CrawlerProcess()

    process.crawl(HouseSpider)

    process.start()

    df = pd.DataFrame(data=spider.get_data())

    with open("python-w-4-d-1-hw-bs-result.xlsx", "wb") as f:
        df.to_excel(f)


class HouseSpider(scrapy.Spider):

    name = "house"

    _data = []

    url = "https://bina.az/"

    headers = {"user-agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36"}

    def start_requests(self):

        yield scrapy.Request(url=HouseSpider.url,
                callback=self.parse, method='GET', headers=self.headers)
    

    def parse(self, response):

        with open("house.html", "wb") as f: f.write(response.body)

        # 2

        links = pd.unique([HouseSpider.url + i for i in response.css("div.items-i.vipped a.item_link::attr(href)").getall()])

        # 1

        locations = response.css("div.items-i.vipped div.card_params div.location::text").getall()[:len(links)]

        # 3

        date_time = [re.sub(r'\w+, ', '', i) for i in
            response.css("div.items-i.vipped div.card_params div.card_footer div.city_when::text").getall()[:len(links)]]

        # 4
 
        price = response.css("div.items-i.vipped div.card_params div.abs_block div.price span.price-val::text").getall()[:len(links)]

        HouseSpider._data = {
                              "location": locations,

                              "link": links,

                              "date_time": date_time,

                              "price": price
                            }

    def get_data(self): return HouseSpider._data


if __name__ == "__main__": main()


