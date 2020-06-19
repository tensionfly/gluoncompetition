import mxnet as mx

def train_net(net,train_data,loss_operation,trainer,lr_scheduler,num_epochs,path_save_model, \
                print_save_model_opertion,test_data,evaluate,print_iterate_operation=None,ctx=mx.cpu()):
    """
    training net
    """
    net.collect_params().reset_ctx(ctx)
    warmup_steps=lr_scheduler.warmup_steps
    
    for e in range(num_epochs):
        assert warmup_steps<=len(train_data),\
        'please set warmup_steps that is not bigger than len(train_data)'

        train_loss=0.

        if e>0:
            trainer.set_learning_rate(lr_scheduler(e+warmup_steps-1))

        for i,(data,label) in enumerate(train_data):

            if e==0:
                if i>=warmup_steps-1:
                    trainer.set_learning_rate(lr_scheduler(warmup_steps-1))
                else:
                    trainer.set_learning_rate(lr_scheduler(i))
            
            data = data.as_in_context(ctx) 
            label = label.as_in_context(ctx)
            
            with mx.autograd.record():
                output=net(data)
                loss=loss_operation(output,label)
                
            loss.backward()
            trainer.step(data.shape[0],ignore_stale_grad=True)
            train_loss+=loss.mean().asscalar()

            if print_iterate_operation is not None:
                print_iterate_operation(e,i,loss,output,label)

        train_loss/=len(train_data)
        evalu_result=evaluate(test_data,net,ctx)
        print_save_model_opertion(e,train_loss,evalu_result,net,path_save_model)

def train_net_kfold(net,train_data_list,loss_operation,trainer,lr_scheduler,num_epochs,path_save_model_list, \
                    print_save_model_opertion,test_data_list,evaluate,print_iterate_operation=None,ctx=mx.cpu()):
    """
    using k fold to train net
    """

    net.collect_params().save('initial.params')

    for train_data,test_data,path_save_model in zip(train_data_list,test_data_list,path_save_model_list):

        net.collect_params().load('initial.params',ctx=ctx)
        train_net(net,train_data,loss_operation,trainer,lr_scheduler,num_epochs,path_save_model, \
                    print_save_model_opertion,test_data,evaluate,print_iterate_operation,ctx)
