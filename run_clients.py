import subprocess

client_processes = []
for i in range(3):
    proc = subprocess.Popen(["python", "client/main.py"])
    client_processes.append(proc)

for proc in client_processes:
    proc.wait()

