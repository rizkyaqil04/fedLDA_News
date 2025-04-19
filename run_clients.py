import subprocess

client_processes = []
for i in range(3):
    proc = subprocess.Popen(["python", "client/client_api.py", str(i)])
    client_processes.append(proc)

for proc in client_processes:
    proc.wait()

