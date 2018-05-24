import numpy as np



class ImagePreproc(object):
    '''Class to handle common image preprocessing (center crops or
    random crops with random flips).
    '''
    
    def __init__(self):
        self.buf = None

    def get_buffer(self, shape, dtype):
        if self.buf is None or self.buf.shape != shape or self.buf.dtype != dtype:
            print 'ImagePreproc: creating new buffer'
            self.buf = np.zeros(shape, dtype)
        return self.buf
        
    def center_crops(self, dat, crop_size):
        '''Returns the center crops.
        dat: (b, 0, 1, c)
        crop_size: e.g. (227,227)
        '''

        nims = dat.shape[0]
        #nch = 3
        nch = dat.shape[-1]
        ret_shape = (nims, crop_size[0], crop_size[1], nch)
        ret = self.get_buffer(ret_shape, dtype=dat.dtype)   # Reuse buffer if possible
        off0 = (dat.shape[1]-crop_size[0])/2
        off1 = (dat.shape[2]-crop_size[1])/2
        ret = dat[:, off0:off0+crop_size[0], off1:off1+crop_size[1], :]
        return ret

    def random_crops(self, dat, crop_size, mirror=True):
        '''Returns random crops of the given size
        dat: (b, 0, 1, c)
        crop_size: e.g. (227,227)
        '''

        nims = dat.shape[0]
        #nch = 3 
        nch = dat.shape[-1]
        ret_shape = (nims, crop_size[0], crop_size[1], nch)
        ret = self.get_buffer(ret_shape, dtype=dat.dtype)   # Reuse buffer if possible
        maxoff0 = dat.shape[1]-crop_size[0]
        maxoff1 = dat.shape[2]-crop_size[1]
        off0s = np.random.randint(0,maxoff0,nims)
        off1s = np.random.randint(0,maxoff1,nims)
        domirror = np.random.randint(0,2,nims)
        for ii in xrange(nims):
            off0 = off0s[ii]
            off1 = off1s[ii]
            if mirror and domirror[ii] == 0:
                ret[ii] = dat[ii, off0:off0+crop_size[0], off1:off1+crop_size[1], :][:,::-1]    # reverse column dimension
            else:
                ret[ii] = dat[ii, off0:off0+crop_size[0], off1:off1+crop_size[1], :]
        return ret


    def color_normalize(self, dat, mean, std):
        '''normalize each color channel with provided mean and std'''
        nims = dat.shape[0]
        nch = 3
        ret_shape = (nims, dat.shape[1], dat.shape[2], nch)
        ret = self.get_buffer(ret_shape, dtype=dat.dtype)   # Reuse buffer if possible
        
        for ii in xrange(nch):
            ret[:,:,:,ii] = (dat[:,:,:,ii] - mean[ii]) / std[ii]
        return ret

