import os
data = open("data.txt","w+")
train = open("train.txt","w+")
test = open("test.txt","w+")

im_dir = "ExtendedYaleB/"
f = os.listdir(im_dir)
folders = [None] * 13
al = 0
bl = 0
for i in range(min(5,len(folders))):
    folders[i] = int(f[i].split("B")[1])
    t = os.listdir(im_dir + f[i])
    for j in range(len(t)):
        imname = im_dir + f[i] + "/" + t[j] + "\t" + str(i + 1) + "\n"
        data.write(imname)
        if j > 500 and j%5 == 0  :
            if("Ambient" not in imname):
                if "info" not in imname:
                    test.write(imname)
                    bl+=1
        elif j < 100:
            if("Ambient" not in imname):
                if "info" not in imname:
                    train.write(imname)
                    al+=1

#    for j in range()

print al,bl
data.close()
train.close()
test.close()
