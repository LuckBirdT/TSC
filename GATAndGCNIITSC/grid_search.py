from itertools import product

# base_cmd = "python run.py -d data/iris.zip -f config/logistic_iris.yaml "
base_cmd = "python gcn_air.py --seed 5 --hidden 256 --dataset citeseer "#不需要搜参的放这

param_grid={
    '--lr':[0.1,0.001,0.0001],
    '--dropout':[0.3,0.5]
    # 'batch_size':[100,200,300],
    # 'nepochs': [100,500,1000]
}

def generate(base_cmd,param_grid):
    keys = param_grid.keys()
    grids = param_grid.values()
    for values in product(*grids):
        param =' '.join([key+' '+str(value) for key,value in (zip(keys,values))])
        # param = param.replace("'",'')
        print(f"{base_cmd} {param}")
# python gcn_air.py --seed 5 --lr 0.05 --hidden 256 --dropout 0.75 --hops 32 --dataset citeseer
if __name__ == '__main__':
    generate(base_cmd,param_grid)