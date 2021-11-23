import argparse
import tensorflow as tf
import timeit
import psutil

from tensorflow.python import training

from utils.common import get_configs

def main():
  tf.keras.backend.set_image_data_format("channels_first")
  parser = argparse.ArgumentParser()
  parser.add_argument("config", nargs="+")
  args, _ = parser.parse_known_args()

  configs = get_configs(args.config[0], is_evaluating=True, restart_training=False)

  print(f'\n==> Loading dataset "{configs.dataset}"')
  dataset = configs.dataset()
  test_dataset = dataset["test"]

  print(f'\n==> Creating model "{configs.model}"')
  model = configs.model()

  @tf.function
  def inference(sample: tf.Tensor):
    return model(sample, training=False)

  x, _ = next(iter(test_dataset))
  single_sample = tf.expand_dims(x[0,:,:], axis=0)
  print("\n\nCalculating inference time and memory usage...")
  print("\nInference Time =", timeit.timeit(lambda: inference(single_sample), number=100) / 100)
  print("RAM Usage =", psutil.virtual_memory()._asdict()['used'] / (1 << 30))

if __name__ == "__main__":
  main()
