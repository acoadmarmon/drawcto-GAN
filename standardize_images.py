import os
import subprocess

for i in os.listdir('./data/images/'):
    if i.split('.')[0] + ".svg" not in os.listdir('./data/processed_images/'):
        return_val = subprocess.run(["python", "linedraw/linedraw.py", "-i", "./data/images/" + i, "-o", "./data/processed_images/{}.svg".format(i.split('.')[0]), "--contour_simplify", "3", "--hatch_size", "512"], capture_output=True)
    if i.split('.')[0] + ".png" not in os.listdir('./data/processed_images/') and i.split('.')[0] + ".svg" in os.listdir('./data/processed_images/'):  
        with open("{}.png".format('./data/processed_images/' + i.split('.')[0]), "w") as f:
            subprocess.call(["rsvg-convert", "-h", "255", "./data/processed_images/{}.svg".format(i.split('.')[0])], stdout=f)

    

