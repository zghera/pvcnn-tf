## Training / Evaluation
* Better to do custom training loop with minimal `tf.keras.Model` (current setup) or subclass `tf.keras.Model` and simply use `compile()`, `fit()`, etc.?
* Is current evaluation scheme of using the training metrics good enough or do I need to have a seperate evaluation method like in the original implementation.

## Layer / Block Definitions
* Should I go back to make sure that all the layers are initialized the same way between Pytorch and TensorFlow as they may be different?
* Confirm: should never be calling SharedMLP with multiple inputs or have mulitple out_channels.

## Custom Ops
* What is the best way to "abstract" ops away in Python if I ever wanted to replace them with a custom C++ tf op in the future? Make another `tf.keras.layers.Layer`, a normal Python function called from another layer's `call()` function (current solution), same thing but with Python function decorated with `@tf.function`?
