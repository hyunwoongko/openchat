def inherit(obj, superclasses):
    obj.__class__ = type(obj.__class__.__name__, superclasses, dict())
