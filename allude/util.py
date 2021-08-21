"""Utils"""

from functools import partial
from operator import mul, methodcaller
from typing import Callable

from i2 import Sig
from lined import Pipe, map_star
from lined.util import func_name
from meshed import DAG
from meshed.util import func_name


class FuncWrap:
    def __init__(self, func):
        self.func = func
        Sig(func)(self)  # make self have func's signature
        self.__name__ = func_name(func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return f"FuncWrap({repr(self.func)})"


# def func_op_obj(*args, op, func, obj, **kwargs):
#     return op(func(*args, **kwargs), obj)


def func_op_obj(op, func, obj):
    obj_op = lambda x: op(x, obj)  # Pattern: Extended partial
    return Pipe(func, obj_op)


def func_op_func(
    op,
    func1,
    func2,
):
    return Pipe(DAG([func1, func2]), methodcaller("values"), map_star(op))


class OperableFunc(FuncWrap):
    def __mul__(self, other):
        if isinstance(other, Callable):
            new_func = func_op_func(op=mul, func1=self.func, func2=other)
            # new_func = partial(func_op_func, op=mul, func1=self.func, func2=other)
        else:
            new_func = func_op_obj(op=mul, func=self.func, obj=other)
            # new_func = partial(func_op_obj, op=mul, func=self.func, obj=other)
        # Sig(self)(new_func)  # give new_func the signature of self

        return OperableFunc(new_func)


def proportional_change(prior, evidence, proportion):
    return prior * (evidence * proportion)


def mk_proportional_change_func(
    prior_argname,
    evidence_argname,
    proportion,
    proportion_argname: str = None,
    name: str = None,
):
    name = name or f"{evidence_argname}_to_{prior_argname}"
    proportion_argname = proportion_argname or f"{prior_argname}_{evidence_argname}"
    func = OperableFunc(partial(proportional_change, proportion=proportion))
    # func = OperableFunc(proportional_change)
    sig = Sig(func)
    # sig = sig.ch_defaults(proportion=proportion)
    sig = sig.ch_names(
        prior=prior_argname,
        evidence=evidence_argname,
        proportion=proportion_argname,
    )

    sig(func)
    func.__name__ = name
    return func
