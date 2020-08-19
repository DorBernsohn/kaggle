 # @author  DBernsohn
import tensorflow as tf
from functools import singledispatch, update_wrapper

def set_TPU():
    """config TPU for tensorflow
    Returns:
        tf object: tf.distribute.TPUStrategy
    """    
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(f"Running on TPU: {tpu.master()}")
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    print(f"REPLICAS: {strategy.num_replicas_in_sync}")
    return strategy

def methdispatch(func):
    """Adjustment of @singledispatchmethod usage to a python version lower than 3.8.
    
    Args:
        func (func): function

    Returns:
        wrapper: function wrapper
    """    
    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper