python -u full_train/train.py --data cora --nlayer 64 --alpha 0.2 --wd1 1e-4 --wd2 1e-4
python -u full_train/train.py --data cora --nlayer 64 --alpha 0.2 --wd1 1e-4 --variant --wd2 1e-4
python -u full_train/train.py --data citeseer --nlayer 64 --wd1 5e-6 --wd2 5e-6 
python -u full_train/train.py --data citeseer --nlayer 64 --wd1 5e-6 --variant --wd2 5e-6 
python -u full_train/train.py --data pubmed --nlayer 64 --alpha 0.1 --wd1 5e-6 --wd2 5e-6 
python -u full_train/train.py --data pubmed --nlayer 64 --alpha 0.1 --wd1 5e-6 --variant --wd2 5e-6 
python -u full_train/train.py --data chameleon --nlayer 8 --lamda 1.5 --alpha 0.2 --wd1 5e-4 --wd2 5e-4
python -u full_train/train.py --data chameleon --nlayer 8 --lamda 1.5 --alpha 0.2 --wd1 5e-4 --variant --wd2 5e-4
python -u full_train/train.py --data cornell --nlayer 16 --lamda 1 --wd1 1e-3 --wd2 1e-3
python -u full_train/train.py --data cornell --nlayer 16 --lamda 1 --wd1 1e-3 --variant --wd2 1e-3
python -u full_train/train.py --data texas --nlayer 32 --lamda 1.5 --wd1 1e-4 --wd2 1e-4
python -u full_train/train.py --data texas --nlayer 32 --lamda 1.5 --wd1 1e-4 --variant --wd2 1e-4
python -u full_train/train.py --data wisconsin --nlayer 16 --lamda 1 --wd1 5e-4 --wd2 5e-4
python -u full_train/train.py --data wisconsin --nlayer 16 --lamda 1 --wd1 5e-4 --variant --wd2 5e-4