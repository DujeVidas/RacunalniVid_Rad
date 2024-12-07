Nnema json fileova (malo su preveliki pa nisan ni pokusaval pushat). Treba skinut sa https://www.kaggle.com/competitions/statoil-iceberg-classifier-challenge/data
test i train


onda se mora pokrenut converters - posto su jsons preveliki treba ih u format u kojem CNN moze lako accesat
pa trainScript
pa evaluate
nez kolko je dobro trebat ce tweakat vjv


u folderu DrugiDataset sam se igral sa drugim datasetom koji je full jednostavnije, treba skinut .npz file  na https://www.kaggle.com/datasets/saurabhbagchi/ship-and-iceberg-images/data

Pokreni svaku skriptu:

    Pokreni load_data.py kako bi provjeril da se podaci ispravno učitavaju.
    Pokreni model.py kako bi potvrdil strukturu modela.
    Pokreni train.py 
    Pokreni validate.py 



VJV ce sve trebat dosta tweakat