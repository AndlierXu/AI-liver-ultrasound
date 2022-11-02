import configparser
import os


class ConfigParser:
    def __init__(self, path):
        if os.path.exists("config/default_local.config"):
            self.local_config = configparser.RawConfigParser()
            self.local_config.read("config/default_local.config")
        else:
            self.local_config = None

        self.default_config = configparser.RawConfigParser()
        self.default_config.read("config/default.config")

        self.config = configparser.RawConfigParser()
        self.config.read(path)

    def get(self, field, name):
        try:
            return self.config.get(field, name)
        except Exception as e:
            try:
                return self.local_config.get(field, name)
            except Exception as e:
                return self.default_config.get(field, name)

    def set(self, sec, field, name):
        try:
            return self.config.set(sec, field, name)
        except Exception as e:
            try:
                return self.local_config.set(sec, field, name)
            except Exception as e:
                return self.default_config.set(sec, field, name)

    def getint(self, field, name):
        try:
            return self.config.getint(field, name)
        except Exception as e:
            try:
                return self.local_config.getint(field, name)
            except Exception as e:
                return self.default_config.getint(field, name)

    def getfloat(self, field, name):
        try:
            return self.config.getfloat(field, name)
        except Exception as e:
            try:
                return self.local_config.getfloat(field, name)
            except Exception as e:
                return self.default_config.getfloat(field, name)

    def getboolean(self, field, name):
        try:
            return self.config.getboolean(field, name)
        except Exception as e:
            try:
                return self.local_config.getboolean(field, name)
            except Exception as e:
                return self.default_config.getboolean(field, name)
