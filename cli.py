import os

import tensorflow as tf

from main import run as m

flags = tf.app.flags

flags.DEFINE_string("visible_gpu", "0", "visible_gpu")

flags.DEFINE_string("datapath", "data/sports.pkl", "datapath")
flags.DEFINE_string("modelpath", "model", "modelpath")

flags.DEFINE_integer("epochs", 1000, "epochs")
flags.DEFINE_string("opt", "Adagrad", "optimizer")
flags.DEFINE_float("lr", 0.1, "lr")
flags.DEFINE_float("norm", 1e-4, "norm")
flags.DEFINE_float("grad_clip", 10.0, "grad_clip")
flags.DEFINE_float("keep_prob", 0.95, "keep_prob")

flags.DEFINE_integer("dim_hidden", 256, "dim_output")
flags.DEFINE_integer("dim_str", 128, "dim_output")
flags.DEFINE_integer("dim_sent", 384, "dim_sent")

flags.DEFINE_integer("beam_width", 10, "beam_width")
flags.DEFINE_integer("batch_size", 8, "batch_size")

flags.DEFINE_string('f', '', 'kernel')

def main(_):
    config = flags.FLAGS
    os.environ["CUDA_VISIBLE_DEVICES"] = config.visible_gpu

    m(config)

if __name__ == "__main__":
    tf.app.run()
    
    
    