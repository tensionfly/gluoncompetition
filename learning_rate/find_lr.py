import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import math

def find_lr(net,trainer,loss_operation,dataiter,lr_begin,lr_end,steps,ctx=mx.cpu()):
    """
    using this function to find a appropriate learing rate by increasing lr step by
    step
    """
    net.collect_params().reset_ctx(ctx)

    lr=[]
    losses=[]

    log_lr_begin=np.log(lr_begin)
    log_lr_end=np.log(lr_end)
    delta_log_lr=(log_lr_end-log_lr_begin)*1.0/steps

    lr_list=np.exp([i*delta_log_lr+log_lr_begin for i in range(steps)])
    
    st_log10_lr=math.floor(math.log(lr_begin,10))
    ed_log10_lr=math.ceil(math.log(lr_end,10))

    lr_list_10=[math.pow(10,x) for x in range(st_log10_lr,ed_log10_lr+1)]
    lr_tuple=tuple(lr_list_10)

    trainer.set_learning_rate(lr_list[0])

    for k,(data,gt) in enumerate(dataiter):

        data=data.as_in_context(ctx)
        gt=gt.as_in_context(ctx)

        with mx.autograd.record():
            pred=net(data)
            loss=loss_operation(pred,gt)

        loss.backward()
        trainer.step(data.shape[0])

        if k<steps-1:
            loss_mean=mx.nd.mean(loss).asscalar()
            print(k,trainer.learning_rate,loss_mean)

            lr.append(trainer.learning_rate)
            losses.append(loss_mean)
            trainer.set_learning_rate(lr_list[k+1])

        if k==steps-1:
            plt.figure()
            plt.xticks(np.log(lr_list_10),lr_tuple)
            plt.xlabel('learning rate')
            plt.ylabel('loss')
            plt.plot(np.log(lr),losses)
            plt.show()

            break