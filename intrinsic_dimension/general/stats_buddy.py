import re
import bisect
from colorama import Style
import numpy as np
import time


class StatsBuddy(object):
    '''Your training stat collecting buddy!

    epochs: epoch E is after seeing and updating from the entire training set E times.
    train_iter: train iter T is after seeing and updating from T train mini-batches.

    Both start at 0, which implies zero training.'''

    def __init__(self, pretty_replaces=None, default_pretty_replaces=True):
        self._epoch = 0
        self._train_iter = 0
        self._wall = None
        self._wall_total = 0      # Saved time, perhaps from other runs
        self._pretty_replaces = []
        if default_pretty_replaces:
            self._pretty_replaces.extend([
                ('train_', ''),
                ('val_', ''),
                ('test_', ''),
                ('loss', 'l'),
                ('accuracy', 'acc'),
                ('cross_entropy','xe'),
                ('cross_ent','xe'),
                ('euclidean','euc'),
                (':0',''),
            ])
        if pretty_replaces:
            self._pretty_replaces.extend(pretty_replaces)

        # Data is stored as dict of four lists: [epoch_list, train_iter_list, weight_list, value_list]
        # Each of the four lists is the same length.
        self._data = {}

    def _get_fetch_kwargs(self, raise_on_empty=True, empty_val=-123, assert_sync_epoch=True):
        return raise_on_empty, empty_val, assert_sync_epoch
 
    @property
    def epoch(self):
        return self._epoch
        
    @property
    def train_iter(self):
        return self._train_iter
        
    def tic(self):
        # Just need to call once on model creation.
        # Must also be called when loading a saved StatsBuddy from disk and resuming run!
        self._wall = time.time()
        
    def toc(self):
        assert self._wall, 'toc called without tic'
        elapsed = time.time() - self._wall
        self._wall_total += elapsed
        self._wall = time.time()
        return self._wall_total
        
    def inc_epoch(self):
        self._epoch += 1

    def inc_train_iter(self):
        self._train_iter += 1
    
    def note(self, **kwargs):
        '''Main stat collection function. See below for methods providing various syntactic sugar.'''
        weight = kwargs['_weight'] if '_weight' in kwargs else 1.0
        for key in sorted(kwargs.keys()):
            if key == '_weight':
                continue
            value = kwargs[key]
            #print key, value
            self.note_one(key, value, weight=weight)
            
    def note_weighted(self, _weight, **kwargs):
        '''Convenience function to call note with explicit weight.'''
        assert '_weight' not in kwargs, 'Provided weight twice (via positional arg and kwarg)'
        self.note(_weight=_weight, **kwargs)

    def note_weighted_list(self, _weight, name_list, value_list, prefix='', suffix=''):
        '''Convenience function to call note with explicit weight and
        list of names and values. Prefix and/or suffix, if given, are
        concatenated to the beginnning and/or end of each name.
        '''
        assert len(name_list) == len(value_list), 'length mismatch'
        for name,value in zip(name_list, value_list):
            final_name = prefix + name + suffix
            self.note_one(final_name, value, weight=_weight)

    def note_list(self, name_list, value_list, prefix='', suffix=''):
        '''Convenience function to call weighted_note_list with a
        weight of 1.0
        '''
        self.note_weighted_list(1.0, name_list, value_list, prefix=prefix, suffix=suffix)

    def note_one(self, key, value, weight=1.0):
        epoch_list, train_iter_list, weight_list, value_list = self._data.setdefault(key, [[], [], [], []])
        epoch_list.append(self.epoch)
        train_iter_list.append(self.train_iter)
        weight_list.append(weight)
        value_list.append(value)
        #print 'Noted: %20s, e: %d, ti: %d, w: %g, v: %g' % (key, self.epoch, self.train_iter, weight, value)

    def last(self, *args, **kwargs):
        '''Get last values as list'''
        raise_on_empty, empty_val, assert_sync_epoch = self._get_fetch_kwargs(**kwargs)
        last_as_dict = self.last_as_dict(*args, raise_on_empty=raise_on_empty, empty_val=empty_val)
        return [last_as_dict[key] for key in args]

    def last_as_dict(self, *args, **kwargs):
        '''Get last values as dict. Not guaranteed for each value to be at the same epoch or training iteration!'''
        raise_on_empty, empty_val, assert_sync_epoch = self._get_fetch_kwargs(**kwargs)
        ret = {}
        for key in args:
            epoch_list, train_iter_list, weight_list, value_list = self._data.setdefault(key, [[], [], [], []])
            if value_list:
                ret[key] = value_list[-1]
            else:
                if raise_on_empty:
                    raise Exception('No value for %s yet recorded' % key)
                else:
                    ret[key] = empty_val
        return ret

    def last_list_re(self, regex, **kwargs):
        ret = []
        for key in sorted(self._data.keys()):
            if re.search(regex, key):
                ret.append((key, self.last(key, **kwargs)[0]))
        return ret

    def last_pretty_re(self, regex, style='', **kwargs):
        keys_values = self.last_list_re(regex, **kwargs)
        return self._summary_pretty_re(keys_values, style=style)
        
    def epoch_mean(self, *args, **kwargs):
        raise_on_empty, empty_val, assert_sync_epoch = self._get_fetch_kwargs(**kwargs)
        means_as_dict = self.epoch_mean_as_dict(*args, raise_on_empty=raise_on_empty, empty_val=empty_val)
        return [means_as_dict[key] for key in args]
        
    def epoch_mean_as_dict(self, *args, **kwargs):
        '''Get mean of each field over most recently recorded epoch,
        as dict. Not guaranteed to be the same epoch for each value
        unless assert_sync_epoch is True.
        '''
        raise_on_empty, empty_val, assert_sync_epoch = self._get_fetch_kwargs(**kwargs)
        ret = {}
        ep = None
        for key in args:
            epoch_list, train_iter_list, weight_list, value_list = self._data.setdefault(key, [[], [], [], []])
            if value_list:
                if ep is None:
                    ep = epoch_list[-1]
                if assert_sync_epoch:
                    assert ep == epoch_list[-1], 'Epoch mismatch between requested epoch means'
                else:
                    ep = epoch_list[-1]
                ep_end = len(epoch_list)
                ep_begin = bisect.bisect_left(epoch_list, ep)
                #print 'Taking epoch mean over %d records' % (ep_end - ep_begin)
                assert ep_begin != ep_end, 'Logic error with bisect_left or data insertion order.'
                values = np.array(value_list[ep_begin:ep_end])
                weights = np.array(weight_list[ep_begin:ep_end])
                
                # remove nan from `values` and `weights` array
                valid_ids = np.where(~np.isnan(values))[0]
                values = values[valid_ids]
                weights = weights[valid_ids]

                if len(valid_ids) == 0:
                    ret[key] = np.nan
                else:
                    weights = weights / float(max(1e-6, weights.sum()))
                    assert len(values.shape) == 1, 'expected vector'
                    assert len(weights.shape) == 1, 'expected vector'
                    ret[key] = np.dot(values, weights)
            else:
                if raise_on_empty:
                    raise Exception('No value for %s yet recorded' % key)
                else:
                    ret[key] = empty_val
        return ret
        
    def epoch_mean_summary_re(self, regex, **kwargs):
        return ', '.join(self.epoch_mean_list_re(regex, **kwargs))

    def epoch_mean_pretty_re(self, regex, style='', **kwargs):
        keys_values = self.epoch_mean_list_re(regex, **kwargs)
        return self._summary_pretty_re(keys_values, style=style)
        
    def epoch_mean_list_re(self, regex, **kwargs):
        ret = []
        for key in sorted(self._data.keys()):
            if re.search(regex, key):
                ret.append((key, self.epoch_mean(key, **kwargs)[0]))
        return ret

    def epoch_mean_list_all(self, **kwargs):
        ret = []
        for key in sorted(self._data.keys()):
            ret.append((key, self.epoch_mean(key, **kwargs)[0]))
        return ret

    def _summary_pretty_re(self, keys_values, style=''):
        '''Produce a short, printable summary. Strips "train_" and "test_" strings assuming they will be printed elsewhere.'''
        ret = []
        losses_seen = 0
        for key, value in keys_values:
            short = key
            for orig,new in self._pretty_replaces:
                short = short.replace(orig, new)
            tup = (short, value)
            if key in ('loss', 'train_loss', 'val_loss', 'test_loss'):
                ret.insert(0, tup)
                losses_seen += 1
            elif 'loss' in key:
                ret.insert(losses_seen, tup)
                losses_seen += 1
            else:
                ret.append(tup)
        if style:
            return ', '.join(['%s: %s%7s%s' % (tt[0], style, '%.4f' % tt[1], Style.RESET_ALL) for tt in ret])
        else:
            return ', '.join(['%s: %.4f' % (tt[0], tt[1]) for tt in ret])

    def data_per_iter(self):
        ret = {}
        for key in self._data.keys():
            ret[key] = {}
            ret[key]['iter'] = np.array(self._data[key][1])
            ret[key]['val'] = np.array(self._data[key][3])
        return ret
