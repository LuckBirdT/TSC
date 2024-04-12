import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='None')
parser.add_argument('--model', type=str, default='None')

#读取参数
args = parser.parse_args()
scan_dir = "result/{}/{}".format(args.data,args.model)
scan_result = "./record/{}".format(args.model)
filename="result_{}".format(args.data)

def save_result(fname,results):
    with open(fname,'a') as fout:
        fout.write('\n'.join(results))

def read_file(fname):
    with open(fname,'r') as fin:
        head = fin.readline().strip()+',file' #skip head
        value = fin.readline().strip() +","+fname
        return (head,value)
def scan_results(directory_path):
    head = None
    results = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                h,value = read_file(file_path)
                if head is None:
                    head = h
                results.append(value)
    if head :
        results.insert(0,head)
        return results
    else:
        return None


if __name__ =="__main__":
    results = scan_results(scan_dir)
    if results is None:
        print(f"Nothing found in {scan_dir}")
    else:
        file_path = os.path.join(scan_result, filename)
        if os.path.exists(scan_result ):
            save_result(file_path,results)
        else:
            os.makedirs(scan_result)
            save_result(file_path, results)
