 # @author  DBernsohn
import tensorflow as tf

def set_TPU():
    """config TPU for tensorflow
    """     
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(f"Running on TPU: {tpu.master()}")
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    print(f"REPLICAS: {strategy.num_replicas_in_sync}")