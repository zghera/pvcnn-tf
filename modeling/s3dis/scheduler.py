"""TODO if we end up using"""
import tensorflow as tf

class AttentionSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """TODO if we end up using
  https://www.tensorflow.org/text/tutorials/transformer#optimizer
  """
  def __init__(self, d_model=128, warmup_steps=4000, eps=1e-8):
    super().__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps
    self.eps = eps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    # tf.print("----------------")
    # tf.print("step = ", step)
    # tf.print("arg1 = ", arg1)
    # tf.print("arg2 = ", arg2)

    lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2) + self.eps
    # tf.print("lr = ", lr)
    # tf.print("----------------")
    return lr


# -------------------------------------------------------------------
# import matplotlib.pyplot as plt

# temp_learning_rate_schedule = AttentionSchedule()
# plt.plot(temp_learning_rate_schedule(tf.range(80000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
# plt.savefig("custom-schedule.png")
