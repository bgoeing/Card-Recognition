import sklearn
import numpy as np

def from_txt_to_npy():
    with open("vectorY1.txt") as f:
        txt = f.read()
        lst = [[int(x) for x in string.split(', ')]
               for string in txt.replace('[','').split(']')[:-1]]
        nparr = np.array(lst)
        np.save('vectorY1.npy',nparr)

if __name__ == '__main__':
    from_txt_to_npy()