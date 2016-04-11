from .Toy_env import MyEnv as Toy_env
from .MG_two_storages_env import MyEnv as MG_two_storages_env
try:
    from .ALE_env import MyEnv as ALE_env
except ImportError:
    pass