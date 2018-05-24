#! /usr/bin/env python

from time import sleep
from datetime import datetime, timedelta
import random
import sys

def snoozer():
    print 'This program sleeps for random durations, periodically waking up to print the time. Runs until reaching its time limit (3 seconds by default).'
    
    if len(sys.argv) > 1:
        maxTime = float(sys.argv[1])
    else:
        maxTime = 3
        
    time0 = datetime.now()
    while True:
        sleep(random.uniform(0, .5))
        diff = datetime.now()-time0
        diffSeconds = diff.seconds + diff.microseconds/1000000.0
        print diffSeconds
        sys.stdout.flush()
        if diffSeconds > maxTime:
            break

snoozer()
