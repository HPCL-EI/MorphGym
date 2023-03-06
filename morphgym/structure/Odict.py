


class ODict(dict):
    "super dict class, attributes and keys are equal"
    def __init__(self, _dict=None, default=None):
        if default is not None:
            self.setdefault(default)

        for k,v in _dict.items():
            if isinstance(v,dict):
                self.__setitem__(k, ODict(v, default))
            else:
                self.__setitem__(k,v)

    def __getattr__(self, key):
        """
        Allow accessing dictionary values as attributes
        :param key:
        :return:
        """
        return self.__getitem__(key)

    def __setattr__(self, key, value):
        """
        Allow assigning attributes to DictConfig
        :param key:
        :param value:
        :return:
        """

        self.__setitem__(key,value)


    def __delattr__(self, key):
        """
        Allow deleting dictionary values as attributes
        :param key:
        :return:
        """
        self.__delitem__(key)
