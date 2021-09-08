"""
The RAM system is used to conveniently create globally temporary values in
any place of a code.

The values to store in a RAM have the below features:
    - Users do not want to **declare it explicitly** in the program, which
        makes the code rather dirty.
    - Users want to **share** it across functions, or even files.
    - Users use it **temporarily**, such as for debugging
    - Users want to **reuse** a group of values several times, while **reset** each
        value in the group before reusing them will add a great overhead to the code.
"""

__global_ram = {}

def ram_write(k, v):
    __global_ram[k] = v


def ram_pop(k):
    return __global_ram.pop(k)


def ram_append(k, v):
    if k not in __global_ram:
        __global_ram[k] = []
    __global_ram[k].append(v)


def ram_inc(k):
    if k not in __global_ram:
        __global_ram[k] = 0
    __global_ram[k] = __global_ram[k] + 1

def ram_read(k):
    return __global_ram[k]


def ram_has(k):
    return k in __global_ram


def flag_name(k):
    return f"RAM_FLAG_{k}"


def ram_set_flag(k):
    ram_write(flag_name(k), True)


def ram_reset_flag(k):
    if ram_has(flag_name(k)):
        ram_pop(flag_name(k))


def ram_has_flag(k, verbose_once=False):
    ret = ram_has(flag_name(k)) and ram_read(flag_name(k)) is True
    if verbose_once and not ram_has_flag(f"VERBOSE_ONCE_{flag_name(k)}"):
        print(f"INFO: check the flag {k}={ret}, the information only occurs once.")
        ram_set_flag(f"VERBOSE_ONCE_{flag_name(k)}")
    return ret


def ram_globalize(name=None):
    if name is None:
        def wrapper(fun):
            if fun.__name__ in __global_ram:
                raise Exception("{} already in ram.".format(fun.__name__))
            __global_ram[fun.__name__] = fun
            return fun
    else:
        def wrapper(fun):
            if name in __global_ram:
                raise Exception("{} already in ram.".format(name))
            __global_ram[name] = fun
            return fun
    return wrapper



# def ram_linear_analyze(k):
    # y = __global_ram[k]
    # x = list(range(len(y)))
    # reg = LinearRegression().fit(np.array(x).reshape(-1, 1),
    #                              np.array(y).reshape(-1, 1))
    # return "y={:4.2f}*x+{:4.2f}".format(cast_item(reg.coef_), cast_item(reg.intercept_))


def ram_reset(prefix=None):
    if prefix is not None:
        to_reset = []
        for key in __global_ram:
            if key.startswith(prefix):
                to_reset.append(key)
        for key in to_reset:
            __global_ram.pop(key)
    else:
        __global_ram.clear()
        
        