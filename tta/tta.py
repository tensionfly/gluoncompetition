import mxnet as mx
import numpy as np

def tta_rotate(data):
    """
    data: mx.nd.array 1,c,h,w
    return: mx.nd.array 4,c,h,w
    """
    ctx=data.context
    data_np=data[0].asnumpy()
    data_list=[np.rot90(data_np,i,(1,2)) for i in range(1,4)]
    data_list.append(data_np)

    data_tta_np=np.stack(data_list,axis=0)
    data_tta=mx.nd.array(data_tta_np,ctx=ctx)

    return data_tta

def tta_filp(data):
    """
    data: mx.nd.array 1,c,h,w
    return: mx.nd.array 4,c,h,w
    """
    data_list=[data.filp(axis=i) for i in range(2,4)]
    data_list.append(data.flip(axis=2).flip(axis=3))
    data_list.append(data)

    return mx.nd.concat(*data_list,dim=0)

def tta_rotate_filp(data):
    """
    data: mx.nd.array 1,c,h,w
    return: mx.nd.array 6,c,h,w
    """
    rotate_data=tta_rotate(data)
    data_list=[data.filp(axis=i) for i in range(2,4)]
    data_list.append(rotate_data)

    return mx.nd.concat(*data_list,dim=0)
