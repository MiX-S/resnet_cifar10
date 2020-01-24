import torch
import time
import numpy as np


def train_optimized(net, X_train, y_train, X_test, y_test, batch_size=256, num_epoch=50, epoch_info_show=10, weight_decay=0):
    """
    Train net on X_train, y_train and compute accuracy on X_test, y_test for each epoch
    Usage of "with torch.no_grad()" prevent from tracking the computation history and save memory
        
    Returns
    -------
    test_accuracy_history : list of len (num_epoch)
    test_loss_history : list of len (num_epoch)
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0e-3, weight_decay=weight_decay)

    
    t = time.time()
    test_accuracy_history = []
    test_loss_history = []

    
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    for epoch in range(1, num_epoch+1):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()
            net.train()

            batch_indexes = order[start_index:start_index+batch_size]

            X_batch = X_train[batch_indexes].to(device)
            y_batch = y_train[batch_indexes].to(device)

            preds = net.forward(X_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()
            
        
        net.eval()
        with torch.no_grad():
            test_preds = net.forward(X_test)
            loss_value = loss(test_preds, y_test).item()
            test_loss_history.append(loss_value)

            accuracy = (test_preds.argmax(dim=1) == y_test).float().mean().item()
            test_accuracy_history.append(accuracy)
        
            if epoch % epoch_info_show == 0:
                print('Train Epoch: {} Time: {} Accuracy: {}, GPU_Mem_alloc: {} GB, GPU_Mem_cashed: {} GB'\
                      .format(epoch, time.strftime("%H:%M:%S", time.gmtime(time.time() - t)), round(accuracy, 3), \
                            round(torch.cuda.memory_allocated() / 1024**3, 3), round(torch.cuda.memory_cached() / 1024**3, 3)))
              
    torch.cuda.empty_cache()
    del net
    return test_accuracy_history, test_loss_history