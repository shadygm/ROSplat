__all__ = ['ROSNodeManager']


def __getattr__(name):
    if name == 'ROSNodeManager':
        from .ros_util import ROSNodeManager

        return ROSNodeManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
