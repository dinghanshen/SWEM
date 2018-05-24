#! /usr/bin/env python

from time import sleep
from datetime import datetime, timedelta
import random
import sys

def sleeper():
    print 'This program sleeps for random durations, periodically waking up to print the time. Runs until killed.'
    
    time0 = datetime.now()
    while True:
        sleep(random.uniform(0, .5))
        diff = datetime.now()-time0
        diffSeconds = diff.seconds + diff.microseconds/1000000.0
        print diffSeconds
        sys.stdout.flush()

sleeper()
