import mxnet as mx

def dice_loss(pred,gt):
    """
    pred: mx.nd.array (n,c,h,w)
    gt: mx.nd.array (n,h,w)

    retrun_loss: mx.nd.array (n,)
    """
    num_classes=pred.shape[1]
    pred=pred.argmax(axis=1)

    pred=pred.one_hot(num_classes)
    gt=gt.one_hot(num_classes)

    inter=(pred*gt).sum(axis=(1,2,3))
    outer=2*(pred.size/(num_classes*pred.shape[0])-inter)+inter
    # outer=gt.size/cfg.num_classes
    return_loss=1.0-inter*(1.0/outer)

    return return_loss

def softmax_celoss_with_ignore(F, label, ignore_label):
    """
    F: mx.nd.array (n,h*w,c)
    label: mx.nd.array (n,h*w)

    return_loss: mx.nd.array (n,)
    """
    output = mx.nd.log_softmax(F,axis=-1)
    label_matrix = mx.nd.zeros(output.shape, ctx=output.context)

    for i in range(label_matrix.shape[2]):
        label_matrix[:,:,i] = (label==i)

    ignore_unit = (label == ignore_label)

    loss = -mx.nd.sum(output * label_matrix, axis=(1,2))
    return loss*(1.0/(output.shape[1] - mx.nd.sum(ignore_unit,axis=1)))

def softmax_celoss_with_weight(pred, gt, weight4cls):
    """
    pred: predicted result
    gt: label
    weight4cls: weight_list eg:[1,4.5,6,3]
    return n,
    """
    pred=-mx.nd.log_softmax(pred)
    pred=pred.reshape((pred.shape[0],pred.shape[1],-1))

    gt_onehot=gt.reshape((gt.shape[0],-1)).one_hot(pred.shape[1])
    gt_onehot=gt_onehot.transpose((0,2,1))
    ratio_array=mx.nd.array(weight4cls,ctx=pred.context).reshape((1,-1,1))

    return (pred*gt_onehot*ratio_array).sum(axis=(1,2))/pred.shape[2]

def gradxy_loss(pred,label):
    """
    pred: mx.nd.array (n,c,h,w)
    label: mx.nd.array (n,c,h,w)

    return_loss: mx.nd.array (n,)
    """

    grad_x_pred=pred[:,:,:,:-1]-pred[:,:,:,1:]
    grad_y_pred=pred[:,:,:-1,:]-pred[:,:,1:,:]

    grad_x_label=label[:,:,:,:-1]-label[:,:,:,1:]
    grad_y_label=label[:,:,:-1,:]-label[:,:,1:,:]

    grad_x_loss=(grad_x_pred-grad_x_label).abs()
    grad_y_loss=(grad_y_pred-grad_y_label).abs()

    grad_loss=grad_x_loss[:,:,:-1,:]+grad_y_loss[:,:,:,:-1]

    return grad_loss.mean(axis=(1,2,3))
