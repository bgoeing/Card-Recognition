import sklearn
import numpy as np

def from_txt_to_npy(source, destination):
    with open(source) as f:
        txt = f.read()
        lst = [[int(x) for x in string.split(', ')]
               for string in txt.replace('[','').split(']')[:-1]]
        nparr = np.array(lst)
        np.save(destination,nparr)

if __name__ == '__main__':
    source = 'vectorY1.txt'
    destination = 'vectorY1.npy'
    from_txt_to_npy(source, destination)