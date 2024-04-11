#训练函数
def train(net, modelName,optimizer, criterion, data):
    # 开启训练模式
    net.train()
    optimizer.zero_grad()
    if modelName=="CGCN" or modelName == "GAT_TSC" or modelName ==  "GCNII_TSC" :
        output,loss_CI,mad,feature_k = net(data.x, data.adj)
    else:
        output,mad,feature_k = net(data.x, data.adj)

    #计算loss
    loss =criterion(output[data.train_mask], data.y[data.train_mask])
    if modelName=="CGCN" or modelName == "GAT_TSC" or modelName ==  "GCNII_TSC":
        loss+=loss_CI

    #计算准确率
    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss, acc,mad,feature_k

#验证和测试
def val_and_test(net, modelName,data):
    net.eval() #开启验证
    if modelName=="CGCN" or modelName == "GAT_TSC" or modelName ==  "GCNII_TSC" :
        output,loss_CI,mad,feature_k = net(data.x, data.adj)
    else:
        output ,mad,feature_k= net(data.x, data.adj)

    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    return  acc_test,mad,feature_k


#准确率计算
def accuracy(output, labels):
    #max(0) 返回每列最大值
    #max(1)返回每行最大值
    preds = output.max(1)[1]
    preds=preds.type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



