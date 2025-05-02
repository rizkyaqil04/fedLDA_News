import subprocess

filenames = ["data1.json", "data2.json", "data3.json"]

for fname in filenames:
    subprocess.Popen(["python", "client/main.py", fname])

