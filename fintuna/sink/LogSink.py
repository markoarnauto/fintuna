
class LogSink:
    """
    A sink for testing.
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, *args):
        print(args)
