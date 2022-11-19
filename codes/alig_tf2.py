"""Implementation of ALI-G in tensorflow 2."""
import tensorflow as tf


class AliG(tf.keras.optimizers.SGD):
    """Optimizer that implements the AliG algorithm.
    """

    def __init__(self, max_lr=None, momentum=0.0, name="AliG", eps=1e-5):
        super(AliG, self).__init__(
            learning_rate=1.0,
            name=name,
            momentum=momentum,
            nesterov=bool(momentum),
          )
        self._set_hyper("max_lr", max_lr if max_lr is not None else 0.0)
        self._set_hyper("eps", eps)

    def minimize(self, loss, var_list, grad_loss=None, name=None, tape=None):
        # first part of method is identical to tf
        grads_and_vars = self._compute_gradients(
            loss, var_list=var_list, tape=tape, grad_loss=grad_loss)

        # compute learning-rate here
        grad_sqrd_norm = sum(tf.norm(grad) ** 2 for grad, _ in grads_and_vars)
        learning_rate = loss / (grad_sqrd_norm + self._get_hyper("eps"))

        max_lr = self._get_hyper("max_lr")

        learning_rate = tf.cond(
            max_lr > 0.0,
            lambda: tf.clip_by_value(
                learning_rate,
                clip_value_min=0,
                clip_value_max=max_lr),
            lambda: learning_rate,
        )

        grads_and_vars = [(g * learning_rate, v) for g, v in grads_and_vars]

        return self.apply_gradients(grads_and_vars, name=name)
