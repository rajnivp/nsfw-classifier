import requests
from threading import Thread
import glob
import os


def create_folders():
    """
    This function will be used to create train and test folders if not already exists
    """
    # Those folders will created to download images if not all ready exists
    train_test_folders = ["category/train/drawings-SFW", "category/train/hentai-NSFW",
                          "category/train/neutral-SFW",
                          "category/train/porn-NSFW", "category/train/sexy-SFW", "category/test/test"]

    for folder in train_test_folders:
        try:
            os.mkdir(folder)

        except FileExistsError:
            print("folder already exists")


def generate_urls(train_size, test_size):
    """
    this function will read data from url text files and generate corresponding list of
    urls for each specific folder and give it specific path and unique names to
    avoid overwriting images and returns list of urls and names
    """
    all_urls = []
    all_names = []
    file_names = {"drawings": "SFW", "hentai": "NSFW", "neutral": "SFW", "porn": "NSFW", "sexy": "SFW"}

    for file in glob.glob("raw_data/*.txt"):
        file_name = file.replace("/", " ").replace("_", " ").replace(".", " ").split()[2]
        if file_name in file_names:
            with open(file) as f:
                drawing_urls = f.readlines()
                all_urls.extend(drawing_urls[:train_size])
                all_urls.extend(drawing_urls[train_size:train_size + test_size])

                train_file_paths = [
                    "category/train/{}-{}/{}_{}.jpg".format(file_name, file_names[file_name], file_name, i)
                    for i in
                    range(train_size)]

                test_file_paths = ["category/test/test/{}_{}.jpg".format(file_name, i) for i in range(test_size)]

                all_names.extend(train_file_paths)
                all_names.extend(test_file_paths)

    return all_urls, all_names


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
    This function will call generate_urls function and pass urls and names list to get_quotes function
    and download images in train and test folder to train and test model
    """
    # Specify number of images for download in train and test folders
    all_urls, all_names = generate_urls(train_size=10, test_size=2)

    # This function will create train and test folders if not all ready exists
    create_folders()

    # Check images urls and names are unique and of same size if True then start downloading
    if len(set(all_urls)) == len(set(all_names)):
        get_quotes(all_urls, all_names)


if __name__ == '__main__':
    main()
