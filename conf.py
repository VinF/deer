import mock
 
MOCK_MODULES = ['numpy', 'scipy', 'matplotlib', 'matplotlib.pyplot', 'scipy.interpolate']
for mod_name in MOCK_MODULES:
sys.modules[mod_name] = mock.Mock()