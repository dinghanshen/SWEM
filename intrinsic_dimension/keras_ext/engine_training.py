#from IPython import embed
import tensorflow as tf
from keras.models import Model
from keras.layers import Input

from general.util import DotDict
from .util import full_static_shape



class ExtendedModel(Model):
    '''Slight extensions of the Keras model class.'''
    
    def __init__(self, input, output, name=None):
        super(ExtendedModel, self).__init__(input, output, name=name)
        self.v = DotDict()
        #self._vars = OrderedDict()
        self._trackable = set()
        self._extra_trainable_weights = []
        self._extra_non_trainable_weights = []

    def add_loss_reg(self):
        '''Adds losses for all attached regularizers'''

        # New Keras interface for regularization / etc layer losses
        losses = []
        for loss in self.losses:
            if loss is None or loss == 0 or loss == 0.0:
                continue
            losses.append(loss)

        if len(losses) > 0:
            print 'Regularizer and other internal losses from model: %d losses' % len(losses)
            for loss in losses:
                print '   loss var=%s' % loss
            self.add_trackable('loss_reg', tf.add_n(losses, name='loss_reg'))
        if 'loss_reg' not in self.v:
            print 'Regularizer and other internal losses from model: none to add.'
        
    def add_var(self, name_or_var, var=None, trackable=False):
        '''Call like self.add_var('name', var) or self.add_var(var) to use var.name as name.'''
        if var is None:
            var = name_or_var
            name = var.name
        else:
            name = name_or_var
        self.v[name] = var
        if trackable:
            self._trackable.add(name)
        elif name in self._trackable:
            self._trackable.remove(name)

    def add_vars(self, names_or_vars, varss=None, trackable=False):
        '''Call with:
         - one list of vars
         - equal length lists of names and vars
         - dict of name: var pairs
        '''
        if isinstance(names_or_vars, dict):
            for name,var in names_or_vars.iteritems():
                self.add_var(name, var, trackable=trackable)
        elif varss is None:
            for var in names_or_vars:
                self.add_var(var, var=None, trackable=trackable)
        else:
            assert len(names_or_vars) == len(varss), 'should be two lists of equal length'
            for name,var in zip(names_or_vars, varss):
                self.add_var(name, var, trackable=trackable)

    def add_trackable(self, name_or_var, var=None):
        self.add_var(name_or_var, var=var, trackable=True)

    def add_trackables(self, names_or_vars, varss=None):
        self.add_vars(names_or_vars, varss=varss, trackable=True)

    def del_var(self, name):
        '''Remove var if it exists'''
        if name in self.v:
            del self.v[name]
            if name in self._trackable:
                self._trackable.remove(name)

    @property
    def var_names(self):
        return self.v.keys()

    @property
    def trackable_names(self):
        return [k for k in self.var_names if k in self._trackable]

    @property
    def vars(self):
        return self.get_vars()

    def get_vars(self, var_names=None):
        if var_names is None:
            var_names = self.var_names
        return [self.v[name] for name in var_names]

    @property
    def tensors(self):
        return self.get_tensors()
            
    def get_tensors(self, tensor_names=None):
        return [vv for vv in self.get_vars(var_names=tensor_names) if isinstance(vv, tf.Tensor)]
    
    @property
    def trackable_vars(self):
        return [self.v[k] for k in self.var_names if k in self._trackable]
    
    @property
    def trackable_dict(self):
        return self.get_tensor_dict(self.trackable_names)

    @property
    def update_dict(self):
        return {'update__%d' % ii: update for ii, update in enumerate(self.updates)}

    @property
    def trackable_and_update_dict(self):
        '''Returns a dict of all trackables and updates. Useful for
        training when you want to fetch all trackables and also ensure
        any updates (e.g. for rolling average BatchNormalization
        layers) are fetched.
        '''

        ret = self.trackable_dict
        ret.update(self.update_dict)
        return ret

    def get_tensor_dict(self, tensor_names=None):
        if tensor_names is None:
            tensor_names = self.var_names
        filtered_names = [nn for nn in tensor_names if isinstance(self.v[nn], tf.Tensor)]
        return {kk:self.v[kk] for kk in filtered_names}

    def print_trainable_warnings(self, graph=None):
        '''Print warnings for any vars marked as trainable in the
        model but not graph, and vice versa. A common case where this
        occurs is in BatchNormalization layers, where internal
        variables are updated but not marked as trainable.
        '''

        if graph is None:
            try:
                graph = tf.python.get_default_graph()
            except AttributeError:
                graph = tf.get_default_graph()

        def tag(name):
            if 'batchnormalization' in name and 'running' in name:
                # Keras 1.2.2
                return ' . '
            elif 'batch_normalization' in name and 'moving' in name:
                # Keras 2+
                return ' . '
            else:
                return '***'
        
        # Check which vars are trainable
        trainable_vars_from_graph = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainable_vars_from_model = self.trainable_weights
    
        in_graph_not_model = set(trainable_vars_from_graph).difference(set(trainable_vars_from_model))
        if in_graph_not_model:
            print 'Warning: the following vars are marked as trainable in the graph but not in model.trainable_weights (typical for BatchNormalization layers. "." if expected, "***" if not):'
            print '\n'.join(['   %4s %s: %s' % (tag(vv.name), vv.name, vv) for vv in in_graph_not_model])
        in_model_not_graph = set(trainable_vars_from_model).difference(set(trainable_vars_from_graph))
        if in_model_not_graph:
            print 'Warning: the following vars are in model.trainable_weights but not marked as trainable in the graph:'
            print '\n'.join(['   %4s %s: %s' % (tag(vv.name), vv.name, vv) for vv in in_model_not_graph])
            
    def add_extra_trainable_weight(self, weight):
        self._extra_trainable_weights.append(weight)

    @property
    def extra_trainable_weights(self):
        return self._extra_trainable_weights

    @property
    def trainable_weights(self):
        tw = super(ExtendedModel, self).trainable_weights
        tw.extend(self.extra_trainable_weights)
        return tw

    def add_extra_non_trainable_weight(self, weight):
        self._extra_non_trainable_weights.append(weight)

    @property
    def extra_non_trainable_weights(self):
        return self._extra_non_trainable_weights

    @property
    def non_trainable_weights(self):
        ntw = super(ExtendedModel, self).non_trainable_weights
        ntw.extend(self.extra_non_trainable_weights)
        return ntw


class LazyModel(ExtendedModel):
    '''Like ExtendedModel. But lazy and nestable.

    In general, we would like to be able to encapsulate functionality
    in larger containers than single layers. However, this is
    difficult because when using the standard Model (and
    ExtendedModel), you must know the input shape in order to make a
    placeholder Input layer. This is far less convenient than, say,
    just being able to call a Dense(123) layer on an input of unknown
    width and having the shape inferred at build time. LazyModel
    solves this problem by delaying the model build until the first
    time it is actually called on a real node in the graph, at which
    point an internal Input layer is constructed on the fly (and
    generally then not used).

    Known issues:

      - BatchNormalization layers fail in mode 0 (because they are
        called twice). Workaround: use in mode 1 or 2 or outside
        LazyModel.

      - Layer activity_regularizers do not work well, because then
        there end up being two copies (one on the activation resulting
        from the internal Input layer). Workaround: use
        activity_regularizers only outside the LazyModel.

      - There still ends up being a dangling tf.placeholder in the
        graph. See notes in exp/model_keras_hacking/ for failed
        more elegant solutions.
    '''

    def __init__(self, model_function):
        self._model_function = model_function
        self._lazy_has_run = False
        # Delay rest of construction until first call

    def __call__(self, inputs, mask=None):
        if not self._lazy_has_run:
            input_was_list_tuple = isinstance(inputs, list) or isinstance(inputs, tuple)
            if input_was_list_tuple:
                input_list = inputs
            else:
                input_list = [inputs]
            # Make short-lived Input Layers for each x this was called with
            input_layers = []
            warn_prefix = 'if_you_get_a_must_feed_placeholder_error_here_it_is_because_you_used_an_activity_regularizer._ask_jason'
            for inp in input_list:
                #ll = Input(tensor=inp, batch_shape=inp._keras_shape, dtype=inp.dtype, name='real_input_from__%s' % inp.name.replace('/','_').replace(':','_'))
                #ll = Input(batch_shape=inp.get_shape().as_list(), dtype=inp.dtype, name='%s.%s' % (warn_prefix, inp.name.replace('/','_').replace(':','_')))
                shape = full_static_shape(inp)
                ll = Input(batch_shape=shape, dtype=inp.dtype, name='%s.%s' % (warn_prefix, inp.name.replace('/','_').replace(':','_')))
                input_layers.append(ll)

            if not input_was_list_tuple:
                input_layers = input_layers[0]

            # Call function of inputs to get output tensors
            # And then initialize the entire model.
            outputs = self._model_function(input_layers)
            super(LazyModel, self).__init__(input_layers, outputs)

            self._lazy_has_run = True

        # Now actually call the model and return the outputs
        return super(LazyModel, self).__call__(inputs, mask=mask)


