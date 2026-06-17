import os

keywords = ["gridlock", "hackerearth", "flipkart", "astram"]
paths = [r"C:\Users\KIIT\Desktop", r"C:\Users\KIIT\Downloads"]
found = []
for p in paths:
    if os.path.exists(p):
        for root, dirs, files in os.walk(p):
            # Avoid deep directories
            if any(x in root.lower() for x in ['node_modules', '.git', 'venv', 'env', '.metadata', 'appdata', 'miniconda', 'xilinx', 'docker', 'idea', 'eclipse']):
                continue
            for f in files:
                for kw in keywords:
                    if kw in f.lower():
                        found.append(os.path.join(root, f))
                        break

print("Found files:")
for f in found:
    print(f)
