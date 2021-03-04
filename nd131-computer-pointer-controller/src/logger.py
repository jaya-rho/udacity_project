#!/usr/bin/env python3

import logging
from logging import Formatter, StreamHandler, getLogger

class Logger:
    def __init__(self, name='logger', level=None):
        self.logger = getLogger(name)
        self.level = level

        # logger level
        self.set_logger_level(self.level)
        # formatter
        self.formatter = Formatter("[%(filename)s:%(lineno)3s - %(funcName)s()] %(message)s")
        # handler for std_out
        self.handler = StreamHandler()
        self.set_handler_level(self.level)
        self.handler.setFormatter(self.formatter)
        # add a handler on a logger
        self.logger.addHandler(self.handler)

    def set_logger_level(self, level=None):
        """ set a level on a logger
        """
        if level == None:
            # if an argument is not given, disable a logger
            self.logger.disabled = True
        else:
            # if an argument is given
            level = level.upper()
            num_level = getattr(logging, level)
            if not isinstance(num_level, int):
                raise ValueError('Invalid log level: %s' % level)
            else:
                self.logger.setLevel(level)

    def set_handler_level(self, level=None):
        """ set a level on a handler
        """
        # if an argument is given
        if not level == None:
            level = level.upper()
            num_level = getattr(logging, level)
            if not isinstance(num_level, int):
                raise ValueError('Invalid log level: %s' % level)
            else:
                self.handler.setLevel(level)

    def get_logger(self, logger_name='logger'):
        """ get a logger object with the specified name
        """
        return getLogger(logger_name)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warn(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
