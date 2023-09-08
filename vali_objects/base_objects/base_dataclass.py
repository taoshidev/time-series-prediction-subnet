from dataclasses import dataclass, fields
from typing import Optional

import numpy as np


@dataclass
class BaseDataClass:

    def __post_init__(self):
        self.schema_integrity_check()

    def schema_integrity_check(self):
        def t_list_check(f, f_t):
            if f_t == int or f_t == float:
                pass
            else:
                raise TypeError(f"The field `{f.name}` was assigned by `{f_t}` instead of int or float")

        for field in fields(type(self)):
            if field.type == Optional[list[float]] \
                    or field.type == list[float] \
                    or field.type == list[int]:
                if getattr(self, field.name) is not None:
                    t_list_check(field, type(getattr(self, field.name)[0]))
            elif field.type == dict[str, list[float]] \
                    or field.type == dict[str, list[int]]:
                if getattr(self, field.name) is not None:
                    if type(getattr(self, field.name)) != dict:
                        raise TypeError(f"The field `{field.name}` was "
                                        f"assigned by `{type(getattr(self, field.name))}` instead of dict")
                    else:
                        f_type = type([value[0] for key, value in getattr(self, field.name).items()][0])
                        k_type = type([key for key, value in getattr(self, field.name).items()][0])
                        t_list_check(field, f_type)
                        if k_type != str:
                            raise TypeError(f"The field `{field.name}` was "
                                            f"assigned by key value `{k_type}` instead of str")
            elif field.type == np:
                pass
            elif not isinstance(getattr(self, field.name), field.type):
                current_type = type(getattr(self, field.name))
                raise TypeError(f"The field `{field.name}` was assigned by `{current_type}` instead of `{field.type}`")