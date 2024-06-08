import pandas as pd


class FileHandler:
    def __init__(self, path: str):
        self.path = path

    def get_data_object(self):
        pass


class CSVFileHandler(FileHandler):
    def get_data_object(self):
        pd.read_csv(self.path)