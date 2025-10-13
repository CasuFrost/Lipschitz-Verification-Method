import os
from utilities import file_to_array
import torch
def search(path,files):
    for d in os.listdir(path):
        if os.path.isdir(path+'/'+str(d)):
            search(path+'/'+str(d),files)
        else:
            file_name=path+'/'+str(d)
            if file_name[-11:]=='MINLOSS.pth':
                files.append(file_name)

def sorted_models(PATH='models'):
    found_files=[]
    search(PATH,found_files)
    models_and_losses=[]
    for p in found_files:
        try:
            checkpoint = torch.load(p)
            min_loss=checkpoint['min_loss']
            models_and_losses.append((p,float(min_loss)))
        except:
            print('file',p,' is corrupt')
    
    return sorted(models_and_losses, key=lambda elemento: elemento[1])

for p in sorted_models()[:15]:
    print(p)