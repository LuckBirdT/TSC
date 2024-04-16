def train(net, modelName,optimizer, criterion, data):
    net.train()
    optimizer.zero_grad()
    if modelName=="TSC_SGC_P" or modelName == "TSC_GCN" or modelName=="TSC_SGC_C" :
        output,loss_CI,mad,feature_k = net(data.x, data.adj)
    else:
        output = net(data)

    loss =criterion(output[data.train_mask], data.y[data.train_mask])
    if modelName=="TSC_SGC_P" or modelName == "TSC_GCN" or modelName=="TSC_SGC_C" :
        loss+=loss_CI

    acc = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if modelName == "TSC_SGC_P" or modelName == "TSC_GCN" or modelName == "TSC_SGC_C":
        return loss, acc,mad,feature_k
    else:
        return loss,acc

def val_and_test(net, modelName,data):
    net.eval()
    if modelName=="TSC_SGC_P" or modelName == "TSC_GCN" or modelName=="TSC_SGC_C" :
        output,loss_CI,mad,feature_k = net(data.x, data.adj)
    else:
        output = net(data)
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    if modelName == "TSC_SGC_P" or modelName == "TSC_GCN" or modelName == "TSC_SGC_C":
        return acc_test,mad,feature_k
    else:
        return acc_test


def accuracy(output, labels):
    preds = output.max(1)[1]
    preds=preds.type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



