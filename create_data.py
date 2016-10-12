from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import math
import re
from PIL import Image

def readfile(file,labelfile,no,ist,slen=1024, overlap=0.5, NFFT=64):

    # load wav
    NSlide = int(math.floor((1-overlap)*slen))
    fs, data = wavfile.read(file)
    data = data[:,0]
    wlen = len(data)
    # load label

    thisfile = open(labelfile, 'r')
    arrlabel = thisfile.readlines()

    # label = ....
    #p=re.compile('[^a-zA-Z&\s]*')
    #p=re.compile('[^\s]*')
    lenlabel = (len(arrlabel))
    dlabel = []
    p = re.compile(r'\s+')
    for i in range(0,lenlabel,1):
        index = p.split(arrlabel[i])
        dlabel.append(index)

            # index format = ['2', '3.425543126', '3.520399563', '0.094856438', 'AF', '']

    #min-max
    outlabel=[]
    i=0
    j=0
    new_value=[]
    for st in range(0,wlen-slen,NSlide):
        new_value.append([])

        en = st+slen-1
        #print(st,en)
        # y = calY(st/fs, en/fs, label)

        start = float(dlabel[i][1])
        end = float(dlabel[i][2])

        stl = float(st/fs)
        enl = float(en/fs)

        if((start>stl and start<enl) or (end>stl and end<enl)):
            if(i<len(dlabel)-1):
                nextstart = float(dlabel[i+1][1])
                if(nextstart<enl and dlabel[i+1][4]=='ES'and dlabel[i][4]=='AF'):
                    i=i+1
                    start = float(dlabel[i][1])
                    end = float(dlabel[i][2])
            outlabel.append(dlabel[i][4])
            #mypath='/Users/Mint/Documents/2_2558/Deep neural network/Shortfile_Pic_C1/Fig/'
            mypath='/Users/Mint/Documents/2_2558/Deep neural network/Shortfile_Pic_C1/Fig_'+dlabel[i][4]+'/'

            #print(dlabel[i][4])
            nextstl = stl+((enl-stl)/2)
            if(end<nextstl and i<len(dlabel)-1):
                i=i+1

        else:
            outlabel.append('NR')
            #mypath='/Users/Mint/Documents/2_2558/Deep neural network/Shortfile_Pic_C1/Fig/'
            mypath='/Users/Mint/Documents/2_2558/Deep neural network/Shortfile_Pic_C1/Fig_NR/'
            if(end<enl and i<len(dlabel)-1):
                i=i+1
        #print(outlabel)
        # X = calX(data[st:en],NFFT)
        t, f, X = signal.spectrogram(data[st:en], fs, nperseg=NFFT, noverlap=(NFFT//2),scaling='spectrum',mode='magnitude')

        LogX = np.log10(X)*(10)
        Logmin= np.amin(LogX)
        Logmax = np.amax(LogX)

        #scale to 0-255
        newmin=0
        newmax=255
        pic= ( (LogX - (Logmin)) / (Logmax - Logmin) ) * (newmax - newmin) + 0
        #print(ist)
           # save image
        im = Image.fromarray(pic.astype(np.uint8))
        #im.save(mypath+str(ist)+'.jpg')
        ist=ist+1
        plt.plot(pic)
        new_value[j].append(( (LogX - (Logmin)) / (Logmax - Logmin) ) * (newmax - newmin) + 0)
        #print(np.shape(new_value))

        j=j+1

    return(outlabel,new_value,ist)

if __name__ == "__main__":
    a=0
    filewav=['C1_31.wav','C1_53.wav','C1_54.wav','C1_55.wav','C1_58.wav','C1_60.wav','C1_61.wav','C1_63.wav','C1_64.wav','C1_66.wav','C1_69.wav','C1_72.wav','C1_73.wav','C1_76.wav','C1_77.wav','C1_79.wav','C1_85.wav','C1_88.wav','C2_10.wav','C2_12.wav','C2_55.wav','C3_3.wav','C3_4.wav','C3_6.wav','C3_13.wav','C3_19.wav','C4_7.wav','C4_11.wav','C4_41.wav','C4_50.wav','C4_54.wav','C5_28.wav','C5_75.wav','C5_80.wav','C5_82.wav','C6_2_t_b.wav','C6_2.wav','C6_3.wav','C6_4_t_a.wav','C6_4_T_b.wav','C6_9.wav']
    filelabel=['C1_31.txt','C1_53.txt','C1_54.txt','C1_55.txt','C1_58.txt','C1_60.txt','C1_61.txt','C1_63.txt','C1_64.txt','C1_66.txt','C1_69.txt','C1_72.txt','C1_73.txt','C1_76.txt','C1_77.txt','C1_79.txt','C1_85.txt','C1_88.txt','C2_10.txt','C2_12.txt','C2_55.txt','C3_3.txt','C3_4.txt','C3_6.txt','C3_13.txt','C3_19.txt','C4_7.txt','C4_11.txt','C4_41.txt','C4_50.txt','C4_54.txt','C5_28.txt','C5_75.txt','C5_80.txt','C5_82.txt','C6_2_t_b.txt','C6_2.txt','C6_3.txt','C6_4_t_a.txt','C6_4_T_b.txt','C6_9.txt']
    #filewav = ['C1_31.wav']
    #filelabel = ['C1_31.txt']
    outlabel=[]
    outputX=[]
    x=1
    while(a<len(filewav)):
        outlabel.append([])
        outputX.append([])
        print(filewav[a])
        mypathwav='/Users/Mint/Documents/2_2558/Deep neural network/es-detector/rawdata/short_file_data/ALL_C/'+filewav[a]
        mypathlabel='/Users/Mint/Documents/2_2558/Deep neural network/es-detector/rawdata/short_file_data/ALL_C/'+filelabel[a]
        outlabel[a],outputX[a],isp =readfile(file=mypathwav,labelfile=mypathlabel,no=a,ist=x)
        x=isp
        a=a+1
        #print(np.shape(outputX[a]))


    #print(np.shape(outputX))
    #print(np.shape(outlabel))