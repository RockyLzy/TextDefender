import argparse


class ProgramArgs:
    # Never use True/False !
    def __init__(self):
        pass

    def _check_args(self):
        assert True

    def __repr__(self):
        basic_ret = ""
        for key, value in self.__dict__.items():
            basic_ret += "\t--{}={}\n".format(key, value)

        deduced_ret = ""
        deduced_args = [ele for ele in dir(self) if ele[0] != '_' and ele not in self.__dict__]
        for key in deduced_args:
            deduced_ret += "\t--{}={}\n".format(key, getattr(self, key))

        ret = "Basic Args:\n" + basic_ret
        if deduced_ret != "":
            ret += "Deduced Args:\n" + deduced_ret
        return ret

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        bool_keys = []
        for key, value in self.__dict__.items():
            # Hack support for true/false
            if isinstance(value, bool):
                bool_keys.append(key)
                value = str(value)
            parser.add_argument('--{}'.format(key),
                                action='store',
                                default=value,
                                type=type(value),
                                dest=str(key))
        parsed_args = parser.parse_args().__dict__
        for ele in bool_keys:
            if parsed_args[ele] in ['True', 'true', 'on', '1', 'yes']:
                parsed_args[ele] = True
            elif parsed_args[ele] in ['False', 'false', 'off', '0', 'no']:
                parsed_args[ele] = False
            else:
                raise Exception('You must pass a boolean value for arg {}'.format(ele))
        self.__dict__.update(parsed_args)
        self._check_args()
        return self
