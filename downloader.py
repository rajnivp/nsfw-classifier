import requests
from threading import Thread
import glob
import os


def create_folders():
    """
    This function will be used to create train and test folders if not already exists
    """
    # Those folders will created to download images if not all ready exists
    train_test_folders = ["category/train/anime-SFW", "category/train/hentai-NSFW", "category/train/neutral-SFW",
                          "category/train/porn-NSFW", "category/train/sexy-SFW", "category/test/test"]

    for folder in train_test_folders:
        try:
            os.mkdir(folder)

        except FileExistsError:
            print("folder already exists")


def generate_urls(train_size, test_size):
    """
    this function will read data from url text files and generate corresponding list of
    urls for each specific folder and given it specific path and unique names to
    avoid overwriting images and returns list of urls and names
    """
    all_urls = []
    all_names = []

    for file in glob.glob("raw_data/*.txt"):
        if "drawing" in file:
            with open(file) as f:
                drawing_urls = f.readlines()
                all_urls.extend(drawing_urls[:train_size])
                all_urls.extend(drawing_urls[train_size:train_size + test_size])

                train_file_paths = ["category/train/anime-SFW/anime_{}.jpg".format(i) for i in range(train_size)]
                test_file_paths = ["category/test/test/anime_{}.jpg".format(i) for i in range(test_size)]

                all_names.extend(train_file_paths)
                all_names.extend(test_file_paths)

        elif "hentai" in file:
            with open(file) as f:
                hentai_urls = f.readlines()
                all_urls.extend(hentai_urls[:train_size])
                all_urls.extend(hentai_urls[train_size:train_size + test_size])

                train_file_paths = ["category/train/hentai-NSFW/hentai_{}.jpg".format(i) for i in range(train_size)]
                test_file_paths = ["category/test/test/hentai_{}.jpg".format(i) for i in range(test_size)]

                all_names.extend(train_file_paths)
                all_names.extend(test_file_paths)

        elif "neutral" in file:
            with open(file) as f:
                neutral_urls = f.readlines()
                all_urls.extend(neutral_urls[:train_size])
                all_urls.extend(neutral_urls[train_size:train_size + test_size])

                train_file_paths = ["category/train/neutral-SFW/neutral_{}.jpg".format(i) for i in range(train_size)]
                test_file_paths = ["category/test/test/neutral_{}.jpg".format(i) for i in range(test_size)]

                all_names.extend(train_file_paths)
                all_names.extend(test_file_paths)

        elif "porn" in file:
            with open(file) as f:
                porn_urls = f.readlines()
                all_urls.extend(porn_urls[:train_size])
                all_urls.extend(porn_urls[train_size:train_size + test_size])

                train_file_paths = ["category/train/porn-NSFW/porn_{}.jpg".format(i) for i in range(train_size)]
                test_file_paths = ["category/test/test/porn_{}.jpg".format(i) for i in range(test_size)]

                all_names.extend(train_file_paths)
                all_names.extend(test_file_paths)

        elif "sexy" in file:
            with open(file) as f:
                sexy_urls = f.readlines()
                all_urls.extend(sexy_urls[:train_size])
                all_urls.extend(sexy_urls[train_size:train_size + test_size])

                train_file_paths = ["category/train/sexy-SFW/sexy_{}.jpg".format(i) for i in range(train_size)]
                test_file_paths = ["category/test/test/sexy_{}.jpg".format(i) for i in range(test_size)]

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
    all_urls, all_names = generate_urls(train_size=1000, test_size=200)

    # This function will create train and test folders if not all ready exists
    create_folders()

    # Check images urls and names are unique and of same size if True then start downloading
    if len(set(all_urls)) == len(set(all_names)):
        get_quotes(all_urls, all_names)


if __name__ == '__main__':
    main()
