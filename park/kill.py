import os


script = "run.py"
cmd = "ps -ef | grep " + script + " | awk \'{print $2}\' | xargs kill -9"
print(cmd)
os.system(cmd)
