import importlib

def _import_module(module_name):
    try:
        return importlib.import_module(f'dpti.workflows.{module_name}')
    except ImportError:
        return importlib.import_module(f'.{module_name}', package='dpti.workflows')

MODULES_TO_IMPORT = [
    'job_executor',
    # 'another_module',
    # 'yet_another_module',
]

# dynamically import sub modules
for module_name in MODULES_TO_IMPORT:
    globals()[module_name] = _import_module(module_name)