import requests
from threading import Thread


# Put text file located in raw_data here to create list of image urls
with open("raw_data/urls_neutral.txt") as file:
    image_urls = file.readlines()


class QuoteGetter(Thread):
    """
    Create QuoteGetter class which will inherit from Thread class
    """
    def __init__(self, url, name):
        super().__init__()
        self.data = None
        self.url = url
        self.name = name

    def run(self):

        try:
            self.data = requests.get(self.url, verify=True)
            with open(self.name, 'wb') as handler:
                handler.write(self.data.content)

        except requests.exceptions.ConnectionError:
            pass


def get_quotes(urls, names):
    """
    Create instances of QuoteGetter class
    and use run method on them to speed up downloading process
    """
    threads = [QuoteGetter(url, name) for url, name in zip(urls, names)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def main():
    """
    Create list of urls based how many you want to download from bigger image_urls list
    Create names list to define folder to store downloaded images and set its names
    Download images in train and test folder to train and test model
    """
    urls = image_urls[:1000]
    names = ["category/train/neutral/neutral_image{}.jpg".format(i) for i in range(1, 1001)]

    # Make sure size of urls and their name given in names list are same to avoid overwriting images
    if len(urls) == len(names):
        get_quotes(urls, names)


if __name__ == '__main__':
    main()
