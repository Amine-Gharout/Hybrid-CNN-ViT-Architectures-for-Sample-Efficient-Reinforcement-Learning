import os, sys, pkgutil
print('cwd:', os.getcwd())
print('sys.path[0]:', sys.path[0])
print('exists envs:', os.path.exists('envs'))
print('listdir[:10]:', os.listdir('.')[:10])
loader = pkgutil.find_loader('envs')
print('pkgutil.find_loader(envs):', loader)
try:
    import envs
    print('import envs succeeded')
    import envs.atari_env as ae
    print('import envs.atari_env succeeded')
except Exception as e:
    import traceback; traceback.print_exc()
