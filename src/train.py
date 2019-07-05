import time
import torch
import torch.nn as nn
import torch.optim as optim


def standard_reporter(report_interval):
    running_loss = [torch.zeros(1, device=device)]
    start = time.time()
    def reporter(loss, iters, epoch, batch_idx):        
        running_loss[0] += loss
        if iters % report_interval == 0:
            print(time.time() - start, float(running_loss[0].cpu()) / report_interval)
            running_loss[0] *= 0

    return reporter
            
            
            
    
    

def train_model(model, epochs, data_loader, device, optimizer=optim.SGD(model.parameters(), lr=args.lr, momentum=0.9), report=standard_reporter(100)):

    lf = nn.CrossEntropyLoss()
    iters = 0
    for epoch in range(epochs):
        for batch_idx, (x, target) in enumerate(data_loader):
            iters += 1
            optimizer.zero_grad()
            if use_cuda:
                x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
                out = model(x)
            
            loss = lf(out, target)
            loss.backward()
            optimizer.step()
            report(loss.detach(), iters, epoch, batch_idx)
