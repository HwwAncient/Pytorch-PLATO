#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
import time


class Timer(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.pass_time = 0.0

    def start(self):
        self.time = time.time()

    def end(self):
        self.pass_time += time.time()  - self.time

    def __enter__(self):
        self.start()

    def __exit__(self, *args, **kwargs):
        self.end()


timers = defaultdict(Timer)
