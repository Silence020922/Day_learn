python -u semi_train/train.py --data cora --nlayer 64 --test
python -u semi_train/train.py --data cora --nlayer 64 --variant --test
python -u semi_train/train.py --data citeseer --nlayer 32 --hideen_size 256 --lamda 0.6 --dropout 0.7 --test
python -u semi_train/train.py --data citeseer --nlayer 32 --hideen_size 256 --lamda 0.6 --dropout 0.7 --variant --test
python -u semi_train/train.py --data pubmed --nlayer 16 --hideen_size 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --test
python -u semi_train/train.py --data pubmed --nlayer 16 --hideen_size 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4 --variant --test