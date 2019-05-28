try:
    from apex.amp import float_function
except ImportError as e:
    def float_function(fn):
        return fn
