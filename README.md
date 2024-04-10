Official pytorch code for  paper ：TSC: A Simple Two-Sided Constraint against Over-Smoothing

## example
python main.py --model TSC_SGC_C --data cora --lr 0.01 --lamda 0.5 --k 1 --tau 0.6 --dropout_c 0.0 --nlayer 16 --epochs 80 --seed 30

python main.py --model TSC_SGC_C --data citeseer --lr 0.01  --lamda 0.5 --k 4 --tau 0.5 --dropout_c 0.0 --nlayer 32 --epochs 80 --seed 5

python main.py --model TSC_SGC_P --data pubmed --lr 0.01 --lamda 0.5 --k 1 --tau 0.1 --dropout_c 0.0 --nlayer 4 --epochs 30 --seed 0

python main.py --model TSC_GCN --lr 0.01  --data cora --tau 0.3 --dropout 0.3 --dropout_c 0.3  --lamda 0.5  --nlayer 16 --epochs 200 --seed 30

 python main.py --model TSC_GCN --lr 0.01  --data citeseer --tau 0.3 --dropout 0.6 --dropout_c 0.6 --nlayer 8 --epochs 200 --seed 5 --lamda 0.4

python main.py --model TSC_GCN --lr 0.01 --data pubmed --tau 0.5 --dropout 0.3 --dropout_c 0.6 --nlayer 4 --lamda 0.4 --epochs 200
