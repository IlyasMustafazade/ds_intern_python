import re

import numpy as np
import pandas as pd
import scrapy
from scrapy.crawler import CrawlerProcess


def main(): 

    spider = RedditSpider()

    process = CrawlerProcess()

    process.crawl(RedditSpider)

    process.start()

    df = pd.DataFrame(data=spider.get_data())

    with open("python-w-4-d-1-cs-result.csv", "wb") as f: df.to_csv(f)


class RedditSpider(scrapy.Spider):

    name = "reddit"

    url = "https://www.reddit.com/r/cpp/"

    _data = []


    def start_requests(self):

        return (scrapy.Request(url=RedditSpider.url, method="GET", callback=self.parse), )


    def parse(self, response):

        # 1
        
        domain = response.css("h3::text").getall()

        # 2

        vote = response.css("div._1rZYMD_4xY3gRcSS3p8ODO._3a2ZHWaih05DgAOtvu6cIo::text").getall()

        # 3

        date = response.css("span._2VF2J19pUIMSLJFky-7PEI::text").getall()

        # 4

        link = [RedditSpider.url[:-7] + i for i in
                   response.css("a.SQnoC3ObvgnGjWt90zD9Z._2INHSNB8V5eaWp4P0rY_mE::attr(href)").getall()]

        RedditSpider._data = {
                                "Domains": domain,

                                "Votes": vote,

                                "Dates": date,

                                "Links": link
                             }


    def get_data(self): return RedditSpider._data


if __name__ == "__main__": main()

