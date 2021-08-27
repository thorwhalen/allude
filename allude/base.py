"""Base functionality"""

from functools import wraps
from i2 import Sig
from inspect import Parameter


def transparent_ingress(*args, **kwargs):
    return args, kwargs


def transparent_egress(output):
    return output


class Wrap:
    def __init__(self, func, ingress=None, egress=None):
        self.func = func
        wraps(func)(self)  # TODO: should we really copy everything by default?

        # remember the actual value of ingress and egress (for reduce to reproduce)
        self._ingress = ingress
        self._egress = egress

        if ingress is not None:
            self.ingress = ingress
            self.__signature__ = Sig(
                ingress, return_annotation=Sig(func).return_annotation
            )
        else:
            self.ingress = transparent_ingress

        if egress is not None:
            self.egress = egress
            ingress_return_annotation = Sig(ingress).return_annotation
            if ingress_return_annotation is not Parameter.empty:
                self.__signature__.return_annotation = ingress_return_annotation
        else:
            self.egress = transparent_egress

        self.__wrapped__ = func
        # TODO: Pros and cons analysis of pointing __wrapped__ to func. partial uses
        #  .func, but wraps looks for __wrapped__

    def __call__(self, *ingress_args, **ingress_kwargs):
        func_args, func_kwargs = self.ingress(*ingress_args, **ingress_kwargs)
        return self.egress(self.func(*func_args, **func_kwargs))

    def __reduce__(self):
        return type(self), (self.func, self._ingress, self._egress)


def wrap(func, ingress=None, egress=None, ingress_factory=None, egress_factory=None):
    return Wrap(func, ingress, egress)


def test_wrap():
    def ingress(a, b: str, c="hi"):
        return (a + len(b) % 2,), dict(string=f"{c} {b}")

    def func(times, string):
        return times * string

    wrapped_func = wrap(func)  # no transformations
    assert wrapped_func(2, "co") == "coco" == func(2, "co")

    wrapped_func = wrap(func, ingress)
    assert wrapped_func(2, "world! ", "Hi") == "Hi world! Hi world! Hi world! "

    wrapped_func = wrap(func, egress=len)
    assert wrapped_func(2, "co") == 4 == len("coco") == len(func(2, "co"))

    wrapped_func = wrap(func, ingress, egress=len)
    assert wrapped_func(2, "world! ", "Hi") == 30 == len("Hi world! Hi world! Hi world! ")

    import pickle

    unpickled_wrapped_func = pickle.loads(pickle.dumps(wrapped_func))
    assert (
        unpickled_wrapped_func(2, "world! ", "Hi")
        == 30
        == len("Hi world! Hi world! Hi world! ")
    )


def items_with_mapped_keys(d: dict, key_mapper):
    for k, v in d.items():
        # key_mapper.get(k, k) will give the new key name if present, else will use the old
        yield key_mapper.get(k, k), v


def invert_map(d: dict):
    new_d = {v: k for k, v in d.items()}
    if len(new_d) == len(d):
        return new_d
    else:
        raise ValueError(f"There are duplicate keys so I can invert map: {d}")


class Ingress:
    def __init__(self, inner_sig, **changes_for_name):
        self.inner_sig = Sig(inner_sig)
        self.outer_sig = self.inner_sig.ch_names(**changes_for_name)
        self.changes_for_name = changes_for_name
        self.old_name_for_new_name = invert_map(changes_for_name)
        self.outer_sig(self)

    def __call__(self, *ingress_args, **ingress_kwargs):
        func_kwargs = self.outer_sig.kwargs_from_args_and_kwargs(
            ingress_args, ingress_kwargs
        )
        func_kwargs = dict(
            items_with_mapped_keys(func_kwargs, self.old_name_for_new_name)
        )
        return self.inner_sig.args_and_kwargs_from_kwargs(func_kwargs)


def mk_ingress_from_name_mapper(foo, name_mapper):
    return Ingress(foo, **name_mapper)


def test_mk_ingress_from_name_mapper():
    from inspect import signature
    import pickle

    def foo(a, b: int, c=7):
        return a + b * c

    name_mapper = dict(a="aa", c="cc")

    ingress = mk_ingress_from_name_mapper(foo, name_mapper)

    foo2 = wrap(foo, ingress)
    assert str(signature(foo2)) == str(signature(ingress)) == "(aa, b: int, cc=7)"
    assert (
        foo(1, 2, c=4) == foo(a=1, b=2, c=4) == foo2(aa=1, b=2, cc=4) == foo2(1, 2, cc=4)
    )

    unpickled_foo2 = pickle.loads(pickle.dumps(foo2))
    assert (
        str(signature(unpickled_foo2)) == str(signature(ingress)) == "(aa, b: int, cc=7)"
    )
    assert (
        foo(1, 2, c=4)
        == foo(a=1, b=2, c=4)
        == unpickled_foo2(aa=1, b=2, cc=4)
        == unpickled_foo2(1, 2, cc=4)
    )
