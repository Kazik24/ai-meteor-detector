import warnings
warnings.simplefilter("ignore",FutureWarning)
import os,datetime,math,sys,io,struct
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.keras.layers import Flatten,Dense,Dropout,InputLayer
from tensorflow.python.keras.layers import Conv2D,GaussianNoise,MaxPool2D


def create_model(input_width,input_height):
    m = tf.keras.models.Sequential()
    m.add(InputLayer(input_shape=(input_width,input_height,6)))
    m.add(Conv2D(filters=32,kernel_size=(4,4),activation='relu'))
    m.add(GaussianNoise(0.01))
    m.add(MaxPool2D(pool_size=(2,2)))
    m.add(Conv2D(filters=48,kernel_size=(4,4),activation='relu'))
    m.add(GaussianNoise(0.01))
    m.add(MaxPool2D(pool_size=(2,2)))
    m.add(Conv2D(filters=64,kernel_size=(4,4),activation='relu'))
    m.add(GaussianNoise(0.01))
    m.add(MaxPool2D(pool_size=(2,2)))
    m.add(Conv2D(filters=64,kernel_size=(4,4),activation='relu'))
    m.add(GaussianNoise(0.01))
    m.add(MaxPool2D(pool_size=(2,2)))
    m.add(Conv2D(filters=64,kernel_size=(4,4),activation='relu'))
    m.add(GaussianNoise(0.01))
    m.add(MaxPool2D(pool_size=(2,2)))
    m.add(Conv2D(filters=64,kernel_size=(4,4),activation='relu'))
    m.add(GaussianNoise(0.01))
    m.add(MaxPool2D(pool_size=(2,2)))
    m.add(Flatten())
    m.add(Dense(128,activation='relu'))
    m.add(Dropout(0.1))
    m.add(Dense(2,activation='softmax'))
    m.compile(optimizer='adam',loss='categorical_crossentropy')
    return m;

def open_image(filename):
    im = Image.open(filename)
    TAG = 36867
    imageTags = im._getexif()
    if imageTags is None or TAG not in imageTags:
        im.close()
        return (im,None)
    date = imageTags[TAG]
    try:
        return (im,datetime.datetime.strptime(date,"%Y:%m:%d %H:%M:%S"))
    except:
        im.close()
        return (im,None)

def scan_timestamps(dirname,nameSuffix,verbose):
    if not isinstance(nameSuffix,list) and not isinstance(nameSuffix,tuple):
        nameSuffix = [nameSuffix]
    imageTimestamp = []
    listedFiles = list(os.listdir(dirname))
    i = 0;
    prevProc = 0;
    for filename in listedFiles:
        i += 1
        curProgress = math.floor((i / len(listedFiles))*10)
        if curProgress != prevProc and verbose:
            prevProc = curProgress
            print("%d%%"%(curProgress*10,))
        filename = os.path.join(dirname,filename)
        if os.path.isfile(filename) and any([filename.lower().endswith(s.lower()) for s in nameSuffix]):
            (im,ts) = open_image(filename)
            if ts is not None:
                imageTimestamp.append((im,ts))
    return imageTimestamp
    
def get_dataset_metadata(dirWithout,dirWith,nameSuffix,verbose):
    if dirWithout is not None:
        if verbose: print("Skanuje folder: %s ..."%(dirWithout,))
        data = scan_timestamps(dirWithout,nameSuffix,verbose)
        data = list([(v[0],v[1],False) for v in data])
    if dirWith is not None:
        if verbose: print("Skanuje folder: %s ..."%(dirWith,))
        withdata = scan_timestamps(dirWith,nameSuffix,verbose)
        data.extend([(v[0],v[1],True) for v in withdata])
    data = sorted(data,key=lambda x:x[1])
    return list(data)

def read_image_data(fileArray,width,height,verbose):
    if verbose: print("Laduję obrazy do pamieci...")
    data = []
    i = 0;
    prevProc = 0;
    for (image,date,label) in fileArray:
        i += 1
        curProgress = math.floor((i / len(fileArray))*10)
        if curProgress != prevProc and verbose:
            prevProc = curProgress
            print("%d%%"%(curProgress*10,))
        filename = image.filename
        image = image.resize((width,height))
        array = np.asarray(image,dtype="float32")
        array = array.transpose(1,0,2)
        image.close()
        array /= 255
        data.append((filename,array,date,label))
    return data

def compose_dataset(imageArray,deltamin,deltamax,verbose):
    if verbose: print("Przygotowuję zbiór danych...")
    x = []
    y = []
    for i in range(len(imageArray)-1):
        rec1 = imageArray[i]
        rec2 = imageArray[i+1]
        delta = (rec2[2] - rec1[2]).total_seconds()
        if delta < deltamax and delta > deltamin and not rec1[3]:
            #if in range and first image is without meteor
            x.append(np.concatenate((rec1[1],rec2[1]),axis=2))
            y.append([0,1] if rec2[3] else [1,0])
    return (np.array(x),np.array(y))

def parse_vals(s,delim):
    arr = s.split(delim)
    if len(arr) != 2: raise ValueError("Zły format, oczekiwano <num>%s<num>"%(delim,))
    return (float(arr[0]),float(arr[1]))
def learn_mode(verbose):
    if len(sys.argv) != 8: raise ValueError("niewlasciwa liczba argumentow")
    (width,height) = parse_vals(sys.argv[4],'x')
    (deltamin,deltamax) = parse_vals(sys.argv[5],'-')
    epochs = int(sys.argv[6])
    outfile = sys.argv[7]
    if deltamin > deltamax: raise ValueError("delta ma zamienione wartosci")
    images = get_dataset_metadata(sys.argv[3],sys.argv[2],['.jpg','.jpeg'],verbose)
    images = read_image_data(images,int(width),int(height),verbose)
    (dsetx,dsety) = compose_dataset(images,deltamin,deltamax,verbose)
    del images #release memory
    if verbose: print("Kompiluje model ...");
    model = create_model(int(width),int(height))
    if verbose: print("Rozpoczęto trenowanie przez {} epok".format(epochs))
    model.fit(dsetx,dsety,batch_size=64,epochs=epochs,validation_split=0.1)
    if verbose: print("Trenowanie zakończone, zapisywanie modelu...")
    del dsetx,dsety #release memory
    buf = io.BytesIO()
    model.save(buf)
    buf = bytearray(buf.getbuffer())
    buf.extend(struct.pack('>f',deltamin))
    buf.extend(struct.pack('>f',deltamax))
    with open(outfile,'wb') as f:
        f.write(buf)
    if verbose: print("Model zapisany do pliku \'%s\'. Wszystko zakończone pomyslnie ^v^"%(outfile,))
    
def select_mode(verbose):
    if len(sys.argv) != 5: raise ValueError("niewlasciwa liczba argumentow")
    if verbose: print("Ładowanie sieci z pliku \'%s\'"%(sys.argv[2],))
    with open(sys.argv[2],'rb') as f:
        arr = f.read()
    stream = io.BytesIO(arr[:-8])
    deltamin = struct.unpack('>f',arr[-8:-4])
    deltamax = struct.unpack('>f',arr[-4:])
    stream.seek(0)
    model = tf.keras.models.load_model(stream)
    del arr,stream
    
    
    
HELP = """
    Użytkowanie:
        {0} <mode> [reszta argumentów]
        
        mode - Tryb pracy skryptu, "learn" pozwala nauczyć sieć neuronową na podanym\
 zbiorze danych. "select" pozwala sklasyfikować folder zdjęc na bazie nauczonego pliku\
 sieci neuronowej.
     Jesli zostal wybrany tryb "learn" to komenda ma postać:
        {0} learn <folder pozytywny> <folder negatywny> <wymiar> <delta> <epoki> <plik sieci>
        Gdzie:
            <folder pozytywny> - scieżka do folderu ze zdjęciami na których występuje\
 poszukiwany obiekt.
            <folder negatywny> - scieżka do folderu ze zdjęciami na których nie ma\
 poszukiwanego obiektu.
            <wymiar> - wymiary zdjęć do których będą konwertowane jako wejcie do sieci.\
 Należy podać wymiary mniejsze od najmniejszego zdjęcia w zbiorze danych, im mniejsze\
 tym szybsze uczenie. Wymiary mają postać <szer>x<wys> np. 1280x720
            <delta> - minimalny i maksymalny czas pomiędzy zdjęciami który klasyfikuje je\
 jako zrobione jedno za drugim. Delta ma postać <min>-<max>, gdzie czasy są podawane w\
 sekundach np. 20-30
            <epoki> - liczba przebiegów trenujących, im większa, tym lepiej nauczy się sieć\
 ale będzie to trwało dłużej.
            <plik sieci> - scieżka do pliku w którym zapisze się wytrenowana sieć z ustawieniami.
    Jesli zostal wybrany tryb "select" to komenda ma postać:
        {0} select <plik sieci> <folder źródłowy> <folder pozytywny>
"""
if __name__ == '__main__':
    try:
        if len(sys.argv) < 2: raise ValueError(HELP.format(os.path.basename(sys.argv[0])))
        mode = str(sys.argv[1]).lower()
        if mode == "learn": learn_mode(True)
        elif mode == "select": select_mode(True)
        else: raise ValueError(HELP.format(os.path.basename(sys.argv[0])))
    except ValueError as e:
        print(e.args[0])
        print(HELP.format(os.path.basename(sys.argv[0])))