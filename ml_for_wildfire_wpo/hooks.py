"""Goes with accum_optimizers.py."""

import keras.backend as K


class Hook(object):
    def __init__(self, module, method, hook=None, recursion=False):
        self.module = module
        self.method = method
        self.hook = hook if hook else self.__hook__
        self.recursion = recursion
        self.__original__ = getattr(module, method)

    def enable(self):
        def hook(*args, **kwargs):
            if hook.called and not self.recursion:
                return self.__original__(*args, **kwargs)

            hook.called = True

            try:
                result = self.hook(*args, **kwargs)
            finally:
                hook.called = False

            return result

        hook.called = False

        setattr(self.module, self.method, hook)

    def disable(self):
        setattr(self.module, self.method, self.__original__)

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable()

    def __hook__(self):
        raise NotImplementedError()


class get_gradients(Hook):
    def __init__(self, optimizer, gradients):
        super(get_gradients, self).__init__(optimizer, 'get_gradients')
        self.gradients = gradients

    def __hook__(self, loss, params):
        return self.gradients


class update_add(Hook):
    def __init__(self, condition, name_scope):
        super(update_add, self).__init__(K, 'update_add')
        self.condition = condition
        self.name_scope = name_scope

    def __hook__(self, x, increment):
        with K.name_scope(self.name_scope):
            if not K.is_tensor(increment):
                increment = K.constant(increment, dtype=K.dtype(x))

            increment = K.switch(self.condition, increment, K.constant(0, dtype=K.dtype(x)))

        return self.__original__(x, increment)


class update_sub(Hook):
    def __init__(self, condition, name_scope):
        super(update_sub, self).__init__(K, 'update_sub')
        self.condition = condition
        self.name_scope = name_scope

    def __hook__(self, x, decrement):
        with K.name_scope(self.name_scope):
            if not K.is_tensor(decrement):
                decrement = K.constant(decrement, dtype=K.dtype(x))

            decrement = K.switch(self.condition, decrement, K.constant(0, dtype=K.dtype(x)))

        return self.__original__(x, decrement)


class update(Hook):
    def __init__(self, condition, name_scope):
        super(update, self).__init__(K, 'update')
        self.condition = condition
        self.name_scope = name_scope

    def __hook__(self, x, new_x):
        with K.name_scope(self.name_scope):
            if not K.is_tensor(new_x):
                new_x = K.constant(new_x, dtype=K.dtype(x))

            new_x = K.switch(self.condition, new_x, x)

        return self.__original__(x, new_x)
