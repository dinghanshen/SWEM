'''Custom Keras regularizers.'''

import keras
import keras.backend as K



class WeightRegularizer(keras.regularizers.WeightRegularizer):
    '''Subclass of Keras WeightRegularizer that doesn't use
    K.in_train_phase, so that total loss can easily be compared
    between train and val modes.
    '''

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.uses_learning_phase = False
        self.p = None

    def get_loss(self):
        loss = 0.0
        if self.l1:
            loss += K.sum(K.abs(self.p)) * self.l1
        if self.l2:
            loss += K.sum(K.square(self.p)) * self.l2
        return loss

class WeightRegularizerMean(keras.regularizers.WeightRegularizer):
    '''Subclass of Keras WeightRegularizer that doesn't use
    K.in_train_phase, so that total loss can easily be compared
    between train and val modes.

    Uses mean instead of sum above.
    '''

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.uses_learning_phase = False
        self.p = None

    def get_loss(self):
        loss = 0.0
        if self.l1:
            loss += K.mean(K.abs(self.p)) * self.l1
        if self.l2:
            loss += K.mean(K.square(self.p)) * self.l2
        return loss


class ActivityRegularizer(keras.regularizers.ActivityRegularizer):
    '''Subclass of Keras ActivityRegularizer that doesn't use
    K.in_train_phase, so that total loss can easily be compared
    between train and val modes.
    '''

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.uses_learning_phase = False
        self.layer = None

    def get_loss(self):
        if self.layer is None:
            raise Exception('Need to call `set_layer` on '
                            'ActivityRegularizer instance '
                            'before calling the instance.')
        loss = 0.0
        for i in range(len(self.layer.inbound_nodes)):
            output = self.layer.get_output_at(i)
            if self.l1:
                loss += K.sum(self.l1 * K.abs(output))
            if self.l2:
                loss += K.sum(self.l2 * K.square(output))
        return loss


def l1(l=0.01):
    return WeightRegularizer(l1=l)


def l2(l=0.01):
    return WeightRegularizer(l2=l)


def l1l2(l1=0.01, l2=0.01):
    return WeightRegularizer(l1=l1, l2=l2)


def activity_l1(l=0.01):
    return ActivityRegularizer(l1=l)


def activity_l2(l=0.01):
    return ActivityRegularizer(l2=l)


def activity_l1l2(l1=0.01, l2=0.01):
    return ActivityRegularizer(l1=l1, l2=l2)
