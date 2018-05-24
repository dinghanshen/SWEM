import os
import errno
import time

class DotDict(dict):
    """
    Example:
    mm = DotDict({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.iteritems():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.iteritems():
                self[k] = v

    def __getattr__(self, attr):
        if not attr in self:
            raise AttributeError(attr)
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, attr):
        if not attr in self:
            raise AttributeError(attr)
        self.__delitem__(attr)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]

    def __repr__(self):
        dict_rep = super(DotDict, self).__repr__()
        return 'DotDict(%s)' % dict_rep


class WithTimer(object):
    def __init__(self, title = '', quiet = False):
        self.title = title
        self.quiet = quiet
        
    def elapsed(self):
        return time.time() - self.wall, time.clock() - self.proc

    def enter(self):
        '''Manually trigger enter'''
        self.__enter__()
    
    def __enter__(self):
        self.proc = time.clock()
        self.wall = time.time()
        return self
        
    def __exit__(self, *args):
        if not self.quiet:
            titlestr = (' ' + self.title) if self.title else ''
            print 'Elapsed%s: wall: %.06f, sys: %.06f' % ((titlestr,) + self.elapsed())


class TicToc(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self._proc = time.clock()
        self._wall = time.time()
        
    def elapsed(self):
        return self.wall(), self.proc()

    def wall(self):
        return time.time() - self._wall

    def proc(self):
        return time.clock() - self._proc


globalTicToc = TicToc()
globalTicToc2 = TicToc()
globalTicToc3 = TicToc()

def tic():
    '''Like Matlab tic/toc for wall time and processor time'''
    globalTicToc.reset()

def toc():
    '''Like Matlab tic/toc for wall time'''
    return globalTicToc.wall()

def tocproc():
    '''Like Matlab tic/toc, but for processor time'''
    return globalTicToc.proc()

def tic2():
    globalTicToc2.reset()
def toc2():
    return globalTicToc2.wall()
def tocproc2():
    return globalTicToc2.proc()
def tic3():
    globalTicToc3.reset()
def toc3():
    return globalTicToc3.wall()
def tocproc3():
    return globalTicToc3.proc()


                
def mkdir_p(path):
    # From https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

