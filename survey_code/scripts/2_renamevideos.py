import sys
from pathlib import Path

try:
    DIRECTORY = sys.argv[1]
except:
    print("Please, specify the directory with the videos to rename")



directory = Path(DIRECTORY)

print(f"======================")
print(f"== FIND DIRECTORIES ==")
print(f"======================")
directories = []
for path in directory.rglob('*'):
    if path.is_dir():
        directories.append(path)
directories = sorted(directories)
for d in directories:
    print(f"Directory: {d}")


output = open("renaming.csv", "w")

print(f"================")
print(f"== FIND FILES ==")
print(f"================")
count = 0
files = []
for directory in directories:
    local_files = []
    for path in (directory).rglob('*'):
        if path.is_file():
            local_files.append(path)
    local_files = sorted(local_files)
    for f in local_files:
        count+=1
        to_write = f"{f}"[len(f"{f.parent.parent}")+1:]
        print(f"File: {to_write}")
        to_rename = f.parent.parent / f"{str(count).zfill(9)}.mp4"
        print(f"Rename: {to_rename}")
        f.rename(to_rename)
        output.write(f"{to_write},{to_rename}\n")
