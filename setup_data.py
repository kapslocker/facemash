import os
# Creates train.txt and test.txt containing filenames and classes for train and test data respectively.
data = open("data.txt","w+")
train = open("train.txt","w+")
test = open("test.txt","w+")
for i in range(1,41):
    im_dir = "orl_faces/s"+str(i)

    l = os.listdir(im_dir)
    for j in range(len(l)):
        data.write(im_dir+"/"+l[j] + "\t" + str(i) + "\n")
        if j>7:
            test.write(im_dir+"/"+l[j] + "\t" + str(i) + "\n")
        else:
            train.write(im_dir+"/"+l[j] + "\t" + str(i) + "\n")

data.close()
train.close()
test.close()
