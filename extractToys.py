import os
import shutil

with open('./data/json/photos.json') as f:
    i = 0
    for line in f:
        filename = line[13:35] 
        source = './data/photos/' + filename + '.jpg'
        dest = './toydata/photos/' + filename + '.jpg'
        shutil.copyfile(source, dest)

        print line,
        i += 1
        if i == 1000:
            break
