from environments.Toy_env import MyEnv as Toy_env
from environments.MG_two_storages_env import MyEnv as MG_two_storages_env
try:
    from environments.ALE_env import MyEnv as ALE_env
except ImportError:
    pass