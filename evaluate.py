import torch
import utils
import numpy as np

####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####


def validate(model, dataset, batch_size=128, n_task=5, test_size=None, verbose=True, allowed_classes_list=None, device='cpu'):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    model = model.to(device)

    precision_track = []
    for t in range(n_task):
        if allowed_classes_list is not None:
            allowed_classes = allowed_classes_list[t]
        with torch.no_grad():
            total_tested = total_correct = 0
            # Set model to eval()-mode
            if hasattr(model, 'encoder'):
                mode = model.classifier.training
                model.encoder.eval()
                model.classifier.eval()
            elif hasattr(model, 'classifier'):
                mode = model.classifier.training
                model.classifier.eval()
            elif hasattr(model, 'heads'):
                mode = model.training
                model.eval()
            elif hasattr(model, 'dc'):
                mode = model.training
                model.eval()
            data_loader = utils.get_data_loader(dataset[t], batch_size)
            for data, labels in data_loader:
                # -break on [test_size] (if "None", full dataset is used)
                if test_size:
                    if total_tested >= test_size:
                        break
                # -evaluate model (if requested, only on [allowed_classes])
                # data, labels = data.to(device), labels.to(device)
                # labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
                
                if hasattr(model, 'encoder'):
                    preds = model.classifier(model.encoder(data.to(device)))
                elif hasattr(model, 'classifier'):
                    preds = model.classifier(data.to(device))
                elif hasattr(model, 'heads'):
                    preds = model(data.to(device))
                elif hasattr(model, 'dc'):
                    preds = model(data.to(device),alpha=0.5)[0]
                if allowed_classes_list is not None:
                    preds = preds[:, allowed_classes]
                    labels = labels-allowed_classes[0]

                predicted = accuracy(preds, labels)[0]
                # -update statistics
                total_correct += predicted
                total_tested += len(data)
        precision = total_correct / total_tested
        precision_track.append(precision)
    # Set model back to its initial mode, print result on screen (if requested) and return it
    if hasattr(model, 'encoder'):
        model.encoder.train(mode=mode)
        model.classifier.train(mode=mode)
    elif hasattr(model, 'classifier'):
        model.classifier.train(mode=mode)
    elif hasattr(model, 'heads'):
        model.train(mode=mode)
    elif hasattr(model, 'dc'):
        model.train(mode=mode)

    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision_track

def validate_one_task(model, dataset, batch_size=128, test_size=None, verbose=True, allowed_classes=None, device='cpu'):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    model = model.to(device)
    data_loader = utils.get_data_loader(dataset, batch_size)
    
    with torch.no_grad():
        total_tested = total_correct = 0
        # Set model to eval()-mode
        if hasattr(model, 'encoder'):
            mode = model.classifier.training
            model.classifier.eval()
            model.encoder.eval()
        elif hasattr(model, 'classifier'):
            mode = model.classifier.training
            model.classifier.eval()
        elif hasattr(model, 'heads'):
            mode = model.training
            model.eval()
        elif hasattr(model, 'dc'):
            mode = model.training
            model.eval()
        for data, labels in data_loader:
            # -break on [test_size] (if "None", full dataset is used)
            if test_size:
                if total_tested >= test_size:
                    break
            # -evaluate model (if requested, only on [allowed_classes])
            # data, labels = data.to(device), labels.to(device)
            # labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
            
            if hasattr(model, 'encoder'):
                preds = model.classifier(model.encoder(data.to(device)))
            elif hasattr(model, 'classifier'):
                preds = model.classifier(data.to(device))
            elif hasattr(model, 'heads'):
                preds = model(data.to(device))
            elif hasattr(model, 'dc'):
                preds = model(data.to(device),alpha=0.5)[0]
            if allowed_classes is not None:
                preds = preds[:, allowed_classes]
                labels = labels-allowed_classes[0]

            predicted = accuracy(preds, labels)[0]
            # -update statistics
            total_correct += predicted
            total_tested += len(data)
    precision = total_correct / total_tested
    # Set model back to its initial mode, print result on screen (if requested) and return it
    if hasattr(model, 'encoder'):
        model.encoder.train(mode=mode)
        model.classifier.train(mode=mode)
    elif hasattr(model, 'classifier'):
        model.classifier.train(mode=mode)
    elif hasattr(model, 'heads'):
        model.train(mode=mode)
    elif hasattr(model, 'dc'):
        model.train(mode=mode)
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision

def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k
    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.
    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return correct_k, batch_size

def validate_office(model, dataset, batch_size=128, test_size=None, verbose=True, allowed_classes_list=None, n_task=0):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Set model to eval()-mode
    mode = model.encoder.training
    model.encoder.eval()
    model.classifier.eval()

    precision_track = []
    for t in range(n_task):
        if allowed_classes_list is not None:
            allowed_classes = allowed_classes_list[t]
        # Loop over batches in [dataset]
        total_tested = total_correct = 0
        for x,y in dataset[t]:
            # -break on [test_size] (if "None", full dataset is used)
            if test_size:
                if total_tested >= test_size:
                    break
            # -evaluate model (if requested, only on [allowed_classes])
            data, labels = x.unsqueeze(0).to(model.device), y.to(model.device)

            with torch.no_grad():
                scores = (torch.nn.functional.softmax(model.forward_base_learner(data),dim=1) 
                                                        if (allowed_classes is None) else 
                                                        torch.nn.functional.softmax(model.forward_base_learner(data),dim=1)[:, allowed_classes]
                                                        )   ##--Revised
                _, predicted = torch.max(scores, 1)
            # -update statistics
            total_correct += (predicted == labels).sum().item()
            total_tested += len(data)
        precision = total_correct / total_tested
        precision_track.append(precision)

        # total_tested = total_correct = 0
        # idx = 0
        # for i in range(1000):
        #     idx,b_size,x,y = dataset.get_batch(idx, i, batch_size)
        #     # -break on [test_size] (if "None", full dataset is used)
        #     if test_size:
        #         if total_tested >= test_size:
        #             break
        #     # -evaluate model (if requested, only on [allowed_classes])
        #     data, labels = x.to(model.device), y.to(model.device)
        #     labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
        #     with torch.no_grad():
                
        #         scores = (torch.nn.functional.softmax(model.forward_base_learner(data),dim=1) 
        #                                                 if (allowed_classes is None) else 
        #                                                 torch.nn.functional.softmax(model.forward_base_learner(data),dim=1)[:, allowed_classes]
        #                                                 )   ##--Revised
        #         _, predicted = torch.max(scores, 1)
        #     # -update statistics
        #     total_correct += (predicted == labels).sum().item()
        #     total_tested += len(data)
        #     # -break on task loop end
        #     if b_size != 128:
        #         break
        # precision = total_correct / total_tested
        # precision_track.append(precision)

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.encoder.train(mode=mode)
    model.classifier.train(mode=mode)
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision_track


def validate_process_adaptation(model, dataset, batch_size=5, from_source=True, verbose=True):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Set model to eval()-mode
    mode = model.encoder.training
    
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=model.is_cuda)
    with torch.no_grad():
        model.encoder.eval()
        model.dc.eval()

        # Loop over batches in [dataset]
        total_tested = total_correct = 0
        for data, _ in data_loader:
            if from_source:
                domain_label = torch.zeros(data.shape[0])
            else:
                domain_label = torch.ones(data.shape[0])
            data, domain_label = data.to(model.device), domain_label.to(model.device)
            
            scores = model.forward_domain_classifier(data, alpha=1.0)
            _, predicted = torch.max(scores, 1)
            # -update statistics
            total_correct += (predicted == domain_label).sum().item()
            total_tested += len(data)
        precision = total_correct / total_tested

    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.encoder.train(mode=mode)
    model.dc.train(mode=mode)
    if verbose:
        print('=> Process adaptation precision: {:.3f}'.format(precision))
    return precision