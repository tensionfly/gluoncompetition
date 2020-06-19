import mxnet as mx

def metric_accuracy(net, dataiter, ctx=mx.cpu()):
    """
    calculate accuracy
    acc=n/N
    """
    acc = mx.metric.Accuracy()

    for data, label in dataiter:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output = net(data)
        output.wait_to_read()
        # predictions = mx.nd.argmax(output, axis=1)

        acc.update(preds=output, labels=label)
    mx.nd.waitall()

    return acc.get()[1]

def metric_f1(net, dataiter, ctx=mx.cpu()):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    precision = true_positives / (true_positives + false_positives)
    recall    = true_positives / (true_positives + false_negatives)
    """
    acc = mx.metric.F1()

    for data, label in dataiter:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output = net(data)
        output.wait_to_read()

        acc.update(preds=output, labels=label)
    mx.nd.waitall()

    return acc.get()[1]

def metric_accuracy_f1(net,dataiter,ctx=mx.cpu()):
    """
    calculate accuracy and f1
    """

    eval_metrics = mx.metric.CompositeEvalMetric()

    for child_metric in [mx.metric.Accuracy(), mx.metric.F1()]:
        eval_metrics.add(child_metric)
    
    for data, label in dataiter:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output = net(data)
        output.wait_to_read()

        eval_metrics.update(preds=output, labels=label)
    mx.nd.waitall()

    return eval_metrics.get()[1]

def metric_mse(net,dataiter,ctx=mx.cpu()):
    """
    mse=∑i(yi−y^i|)**2/n
    """

    mse=mx.metric.MSE()

    for data, label in dataiter:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output = net(data)
        output.wait_to_read()

        mse.update(preds=output, labels=label)
    mx.nd.waitall()

    return mse.get()[1]

def metric_mae(net,dataiter,ctx=mx.cpu()):
    """
    mae=∑i|yi−y^i|/n
    """

    mae=mx.metric.MAE()

    for data, label in dataiter:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output = net(data)
        output.wait_to_read()

        mae.update(preds=output, labels=label)
    mx.nd.waitall()

    return mae.get()[1]

def metric_confusion_matrix(net,dataiter,num_classes,ctx=mx.cpu()):
    """
    calculate confusion_matrix
    """

    confusion_matrix=mx.nd.zeros((num_classes,num_classes),ctx=ctx)

    for data, label in dataiter:
    
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        output=net(data)
        assert num_classes==output.shape[1],'The given num_classes does not match the real num_classes'
        output=output.argmax(axis=1)

        label=label.reshape((-1,)).asnumpy().astype('int32')
        output=output.reshape((-1,)).asnumpy().astype('int32')
        
        for i,j in zip(label,output):
            confusion_matrix[i,j]+=1
    
    sum_confusion_matrix=confusion_matrix.sum(axis=1).reshape((-1,1))

    return confusion_matrix*(1.0/sum_confusion_matrix),confusion_matrix


