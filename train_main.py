import argparse
import torch
import os
import time
import numpy as np
import evaluate
import argparse
import random
from config import *
import copy
from data import get_subset_office31, get_multitask_experiment, get_subset_officeHome
from loggers.exp_logger import MultiLogger
from networks import tvmodels, allmodels, set_tvmodel_head_var
from utils import print_summary, get_data_loader
from networks.clamp import get_model

import pickle
import importlib
import warnings

def seed_everything(seed=0):
    
    cudnn_deterministic = True
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic

def run(s, t, parser, args, iRound, param_stamp,verbose=False):

    #-------------------------------------------------------------------------------------------------#

    # #---------------------#
    # #----- REPORTING -----#
    # #---------------------#
    
    # Task incremental scenrio does not activate memory approach
    scenario = args.scenario
    if args.scenario == 'task':
        if args.singlehead:
            scenario = 'domain'

    #-------------------------------------------------------------------------------------------------#

    #------------------------------#
    #----- MODEL (CLASSIFIER) -----#
    #------------------------------#

    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr
    from approach.continuous_da import Continous_DA_Appr
    from approach.domain_adaptation import dom_adapt_Appr
    Appr = getattr(importlib.import_module(name='approach.' + args.approach), 'Appr')
    
    _, extra_args = parser.parse_known_args(None)
    if issubclass(Appr, Inc_Learning_Appr): # TODO params change for different strategies
        base_kwargs = dict(nepochs=args.epoch, lr=args.lr1, lr_min=args.lr_min, lr_factor=args.lr_factor,
                        lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                        wd=args.wd, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                        wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train)
    elif issubclass(Appr,Continous_DA_Appr):
        base_kwargs = dict(nepochs=args.epoch, pre_epochs=args.warmup_nepochs, 
                        meta_updates=args.epochIn, base_updates=args.epochOut,
                        lr=args.lr1, lr_min=args.lr_min, lr_factor=args.lr_factor,
                        lr_a=args.lr2,
                        lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                        wd=args.wd, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                        wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn, eval_on_train=args.eval_on_train,
                        pseudo=args.pseudo, meta=args.meta, domain_inv=args.domain, alpha=args.alpha)
    elif issubclass(Appr,dom_adapt_Appr):
        base_kwargs = dict(nepochs=args.epoch, lr=args.lr1, lr_min=args.lr_min, lr_factor=args.lr_factor,
                        lr_patience=args.lr_patience, clipgrad=args.clipping, momentum=args.momentum,
                        wd=args.wd, multi_softmax=args.multi_softmax, wu_nepochs=args.warmup_nepochs,
                        wu_lr_factor=args.warmup_lr_factor, fix_bn=args.fix_bn)

    # assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print('Approach arguments =')
    for arg in np.sort(list(vars(appr_args).keys())):
        print('\t' + arg + ':', getattr(appr_args, arg))
    print('=' * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset
    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(extra_args)
        print('Exemplars dataset arguments =')
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print('\t' + arg + ':', getattr(appr_exemplars_dataset_args, arg))
        print('=' * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    # Args -- GridSearch
    if args.gridsearch_tasks > 0:
        from gridsearch import GridSearch
        gs_args, extra_args = GridSearch.extra_parser(extra_args)
        Appr_finetuning = getattr(importlib.import_module(name='approach.finetuning'), 'Appr')
        assert issubclass(Appr_finetuning, Inc_Learning_Appr)
        GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()
        print('GridSearch arguments =')
        for arg in np.sort(list(vars(gs_args).keys())):
            print('\t' + arg + ':', getattr(gs_args, arg))
        print('=' * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(' '.join(extra_args))
    ####################################################################################################################
    
    TypeAblation  = '{}2{}'.format(s, t)
    r_dir = os.path.join(args.r_dir, TypeAblation)

    # Log all arguments
    full_exp_name = s+'2'+t
    full_exp_name += '_' + args.approach
    # if args.exp_name is not None:
    #     full_exp_name += '_' + args.exp_name
    logger = MultiLogger(r_dir, full_exp_name, loggers=args.log, save_models=args.save_models)
    logger.log_args(argparse.Namespace(**args.__dict__, 
    **appr_args.__dict__, **appr_exemplars_dataset_args.__dict__
    ))

    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- Data -----#
    #----------------#

    # Loaders
    seed_everything(seed=2)
    print('Preparing the data...')
    if args.source == DatasetsType.office31:
        src_tr = get_subset_office31(s,args.scenario).train_datasets
        src_te = get_subset_office31(s,args.scenario).test_datasets
        tgt_tr = get_subset_office31(t,args.scenario).train_datasets
        tgt_te = get_subset_office31(t,args.scenario).test_datasets
        config = {'size': 224, 'channels': 3, 'classes': 31}
        classes_per_task = [6,6,6,6,7]
        args.tasks = 5
        transform = src_tr[0].transforms
    elif args.source == DatasetsType.officehome:
        src_tr = get_subset_officeHome(s,args.scenario).train_datasets
        src_te = get_subset_officeHome(s,args.scenario).test_datasets
        tgt_tr = get_subset_officeHome(t,args.scenario).train_datasets
        tgt_te = get_subset_officeHome(t,args.scenario).test_datasets
        config = {'size': 224, 'channels': 3, 'classes': 65}
        classes_per_task = 5
        args.tasks = 13
        transform = src_tr[0].transforms
    elif args.source == "splitMNIST" or args.source == "splitUSPS":
        (src_tr, src_te), config, classes_per_task, transform = get_multitask_experiment(
            name=args.source, scenario=args.scenario, tasks=args.tasks, data_dir=args.d_dir,    
            verbose=True, exception=True if args.seed==0 else False,)
        (tgt_tr, tgt_te), config, classes_per_task, transform = get_multitask_experiment(
            name=args.target, scenario=args.scenario, tasks=args.tasks, data_dir=args.d_dir,    
            verbose=True, exception=True if args.seed==0 else False,)
    else:
        warnings.warn("Warning: %s dataset is not available." % args.source)
    max_task = args.tasks if args.stop_at_task == 0 else min(args.stop_at_task,args.tasks)
    taskcla = [(t,classes_per_task) for t in range(args.tasks)] if classes_per_task is not list \
        else [(t,classes_per_task[t]) for t in range(args.tasks)]
    print(transform)

    # # balance datasets
    # from data import balance_dataset
    # src_tr = balance_dataset(src_tr)
    # tgt_tr = balance_dataset(tgt_tr)

    #-------------------------------------------------------------------------------------------------#

    # Args -- Network                           
    if args.approach ==  'clamp':
        seed_everything(seed=args.seed+iRound)
        # Define main model (i.e., classifier, if requested with feedback connections)
        from networks.clamp import CLAMP
        seed_everything(seed=args.seed+iRound)
        net = CLAMP(dataset_name=args.target, network_name=args.network, \
                num_classes=config["classes"], bottleneck_dim=args.fc_units, \
                scratch=args.scratch, no_pool=args.no_pool, \
                pretrained=args.pretrained
                )
        seed_everything(seed=args.seed+iRound)
        appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
        if Appr_ExemplarsDataset:
            appr_exemplars_dataset_args.__dict__['num_exemplars'] = args.num_exemplars1
            appr_exemplars_dataset_args.__dict__['num_exemplars_per_class'] = args.num_exemplars_per_class1
            appr_kwargs['exemplars_dataset1'] = Appr_ExemplarsDataset(**appr_exemplars_dataset_args.__dict__)
            appr_exemplars_dataset_args.__dict__['num_exemplars'] = args.num_exemplars2
            appr_exemplars_dataset_args.__dict__['num_exemplars_per_class'] = args.num_exemplars_per_class2
            appr_kwargs['exemplars_dataset2'] = Appr_ExemplarsDataset(**appr_exemplars_dataset_args.__dict__)
    elif args.approach ==  'dann':
        from networks.dann import Dann
        seed_everything(seed=args.seed+iRound)
        init_model = get_model(args.network, pretrain=args.pretrained)
        seed_everything(seed=args.seed+iRound)
        net = Dann(init_model,f=args.fc_units,h=int(args.fc_units/2), n_outputs=config["classes"])
        appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
        if Appr_ExemplarsDataset:
            appr_exemplars_dataset_args.__dict__['num_exemplars'] = args.num_exemplars1
            appr_exemplars_dataset_args.__dict__['num_exemplars_per_class'] = args.num_exemplars_per_class1
            appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(**appr_exemplars_dataset_args.__dict__)
    else:
        from networks.network import LLL_Net
        if args.network in tvmodels:  # torchvision models
            tvnet = getattr(importlib.import_module(name='torchvision.models'), args.network)
            if args.network == 'googlenet':
                init_model = tvnet(pretrained=args.pretrained, aux_logits=False)
            else:
                init_model = tvnet(pretrained=args.pretrained)
            set_tvmodel_head_var(init_model)
        else:  # other models declared in networks package's init
            args.network = 'LeNet' if args.network=='lenet' else args.network
            net = getattr(importlib.import_module(name='networks'), args.network)
            # WARNING: fixed to pretrained False for other model (non-torchvision)
            init_model = net(pretrained=False)

        # Network and Approach instances
        seed_everything(seed=args.seed+iRound)
        net = LLL_Net(init_model, config["classes"])

        seed_everything(seed=args.seed+iRound)
        # taking transformations and class indices from first train dataset
        transform = transform
        class_indices = list(range(classes_per_task)) if (type(classes_per_task) is not list) else list(range(classes_per_task[0]))
        appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
        if Appr_ExemplarsDataset:
            appr_exemplars_dataset_args.__dict__['num_exemplars'] = args.num_exemplars1
            appr_exemplars_dataset_args.__dict__['num_exemplars_per_class'] = args.num_exemplars_per_class1
            appr_kwargs['exemplars_dataset'] = Appr_ExemplarsDataset(**appr_exemplars_dataset_args.__dict__)

    # taking transformations and class indices from first train dataset
    if scenario=="class":
        if isinstance(classes_per_task,int):
            classes_list = [list(range(classes_per_task*(i+1))) for i in range(args.tasks)] 
        elif isinstance(classes_per_task,list): 
            classes_list = [list(range(sum(classes_per_task[:i+1]))) for i in range(len(classes_per_task))]   
    elif scenario=="task":
        if isinstance(classes_per_task,int):
            classes_list = [list(range(classes_per_task*i, classes_per_task*(i+1))) for i in range(args.tasks)] 
        elif isinstance(classes_per_task,list): 
            classes_list = [list(range(sum(classes_per_task[:i]), sum(classes_per_task[:i+1]))) for i in range(args.tasks)] 
    elif scenario=="domain":
        classes_list=None

    seed_everything(seed=args.seed+iRound)
    appr = Appr(net, args.device, **appr_kwargs)
    appr.batch_size = args.batch
    if args.approach=='clamp':
        appr.batch_d = args.batch_d
    if args.approach=='dann':
        appr.save_name = 'dann_{}2{}'.format(s, t)
    
    # exemplar existence
    if issubclass(Appr,Continous_DA_Appr):
        if appr.exemplars_dataset1._is_active() or \
            appr.exemplars_dataset2._is_active():
            scenario = "class"
    else:
        if appr.exemplars_dataset._is_active():
            scenario = "class"

    # GridSearch
    if args.gridsearch_tasks > 0:
        ft_kwargs = {**base_kwargs, **dict(logger=logger,
                                           exemplars_dataset=GridSearch_ExemplarsDataset(transform, class_indices))}
        appr_ft = Appr_finetuning(net, args.device, **ft_kwargs)
        gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                                gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)

    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- visualize -----#
    #---------------------#

    if args.analysis:
        # t-SNE plots before alignment
        from copy import deepcopy
        # colors = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494"]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        if issubclass(Appr,Continous_DA_Appr):
            feature_extractor = deepcopy(net.classifier).to(args.device)
            m_type = 1
        elif issubclass(Appr, dom_adapt_Appr):
            feature_extractor = deepcopy(net.encoder).to(args.device)
            m_type = 2
        elif issubclass(Appr, Inc_Learning_Appr):
            feature_extractor = deepcopy(net.model).to(args.device)
            m_type = 2
        tsne_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.p_dir[2:])
        tsne_file   = os.path.join(tsne_dir,  param_stamp + 'tSNE {} {}2{} {} init.png'.format((args.approach).upper(), s, t, args.network))
        
        # if not os.path.exists(tsne_file):
        from utils import visualize
        visualize(src_te,tgt_te,feature_extractor,args.tasks,
                    filename=tsne_file, colors=colors,
                    batch_size=args.batch,device=args.device, m_type=m_type)


    #-------------------------------------------------------------------------------------------------#

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    print('Total tasks : ', args.tasks)
    ##The total number of tasks
    tasktotalnum = args.tasks

    if verbose:
        print("\nTraining...")
    # Keep track of training-time
    start = time.time()
    ##Random model on the Each testing data of each task
    accuracyTiPrecVec = []
    siiPrecVec = []
    sminusiiPrecVec = []  #si-1i
    backwardTransferSTiPrecVec = []
    forwardTransferSTiPrecVec  = []
    indices_trace = []
    # M_S = [] # source sample replay
    # # Set model in training-mode
    # # model.set_train()

    if issubclass(Appr, Inc_Learning_Appr):
        if isinstance(classes_per_task,int):
            net.task_cls =  torch.tensor([classes_per_task]*tasktotalnum)
        elif isinstance(classes_per_task,list): 
            net.task_cls =  torch.tensor(classes_per_task)
        if scenario == "class":
            net.task_offset = torch.cat([torch.LongTensor(1).zero_()] * args.tasks)
        elif scenario == "task":
            net.task_offset = torch.cat([torch.LongTensor(1).zero_(), net.task_cls.cumsum(0)[:-1]])

    print("\n\n--> Random Initial model Evaluation - ({}-incremental learning scenario):".format(args.scenario))
    # Evaluate precision of random initial model on full test-set (testing data of each task)
    # on target
    initialmodeAccuracyPrecVec_tgt = evaluate.validate(
            copy.deepcopy(net), copy.deepcopy(tgt_te), n_task=tasktotalnum, verbose=False, test_size=None,
            # allowed_classes_list=classes_list
        )
    print("\n Initial model Precision on target test-set (softmax classification):")
    for i in range(tasktotalnum):
        print(" - Task {}: {:.4f}".format(i + 1, initialmodeAccuracyPrecVec_tgt[i]))

    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))

    # Loop over all tasks.
    for task in range(args.tasks):  ### Task index starts from 0
        # Early stop tasks if flag
        if (task+1) > max_task:
            continue
        print('#'*108)
        print('\n\nTask {}'.format(task))
        print('#'*108)

        # Find [active_classes]
        active_classes = None  # -> for Domain-IL scenario, always all classes are active
        if scenario == "class":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            if isinstance(classes_per_task,list):
                active_classes = list(range(sum(classes_per_task[:task+1])))
            elif isinstance(classes_per_task,int):
                active_classes = list(range(classes_per_task * (task + 1)))
        elif scenario == "task":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            if isinstance(classes_per_task,list):
                active_classes = list(range(sum(classes_per_task[:(task)]),sum(classes_per_task[:(task+1)])))
            elif isinstance(classes_per_task,int):
                active_classes = list(range(classes_per_task * task, classes_per_task * (task + 1))) 
        elif scenario == "domain":
            active_classes = None
        
        # # Add head for current task
        # if issubclass(Appr, Inc_Learning_Appr):
        #     net.add_head(classes_per_task) if classes_per_task is not list \
        #         else net.add_head(classes_per_task[task])
        # elif issubclass(Appr, Continous_DA_Appr):
        #     net.task_cls = torch.tensor([len(i) for i in classes_list[:(task+1)]])
        #     appr.active_classes = active_classes
        #     appr.classes_per_task = classes_per_task[task] if isinstance(classes_per_task,list) else classes_per_task
        #     appr.source = args.source
        #     print(active_classes)
        # net.task_cls = torch.tensor([len(i) for i in classes_list[:(task+1)]])
        appr.active_classes = active_classes
        appr.classes_per_task = classes_per_task[task] if isinstance(classes_per_task,list) else classes_per_task
        appr.source = args.source
        net.to(args.device)
        if args.approach=='clamp':
            # torch.nn.DataParallel(net, device_ids=[0,1,2,3])
            appr.analysis = args.analysis
        print(active_classes)

        # GridSearch
        if task < args.gridsearch_tasks:

            # Search for best finetuning learning rate -- Maximal Plasticity Search
            print('LR GridSearch')
            # best_ft_acc, best_ft_lr = gridsearch.search_lr(appr.model, task, tgt_tr[task], tgt_te[task])
            best_ft_acc, best_ft_lr = gridsearch.search_lr(appr.model, task, src_tr[task], src_te[task])
            # Apply to approach
            appr.lr = best_ft_lr
            gen_params = gridsearch.gs_config.get_params('general')
            for k, v in gen_params.items():
                if not isinstance(v, list):
                    setattr(appr, k, v)

            # Search for best forgetting/intransigence tradeoff -- Stability Decay
            print('Trade-off GridSearch')
            # best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(args.approach, appr,
            #                                                           task, tgt_tr[task], tgt_te[task], best_ft_acc)
            best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(args.approach, appr,
                                                                      task, src_tr[task], src_te[task], best_ft_acc)
            # Apply to approach
            if tradeoff_name is not None:
                setattr(appr, tradeoff_name, best_tradeoff)

            print('-' * 108)

        # Train
        if issubclass(Appr, Inc_Learning_Appr):
            appr.train(task, src_tr[task], tgt_te[task])
        elif issubclass(Appr, Continous_DA_Appr) or issubclass(Appr, dom_adapt_Appr):
            appr.train(task, src_tr[task], src_te, tgt_tr[task], tgt_te)
        print('-' * 108)
        # Test
        for u in range(task + 1):
            # ************ line 201 - 225 in clamp_.py
            if issubclass(Appr, Inc_Learning_Appr):
                val_loader = get_data_loader(tgt_te[u], args.batch, cuda=True if args.device=='cuda' else False)
            elif issubclass(Appr,Continous_DA_Appr):
                val_loader = tgt_te[u]
            elif issubclass(Appr, dom_adapt_Appr):
                val_loader = get_data_loader(tgt_te[u], args.batch, cuda=True if args.device=='cuda' else False)
            # appr.active_classes = 
            # appr.classes_per_task = 
            test_loss, acc_taw[task, u], acc_tag[task, u] = appr.eval(u, val_loader)
            if u < task:
                forg_taw[task, u] = acc_taw[:task, u].max(0) - acc_taw[task, u]
                forg_tag[task, u] = acc_tag[:task, u].max(0) - acc_tag[task, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[task, u], 100 * forg_taw[task, u],
                                                                 100 * acc_tag[task, u], 100 * forg_tag[task, u]))
            logger.log_scalar(task=task, iter=u, name='loss', group='test', value=test_loss)
            logger.log_scalar(task=task, iter=u, name='acc_taw', group='test', value=100 * acc_taw[task, u])
            logger.log_scalar(task=task, iter=u, name='acc_tag', group='test', value=100 * acc_tag[task, u])
            logger.log_scalar(task=task, iter=u, name='forg_taw', group='test', value=100 * forg_taw[task, u])
            logger.log_scalar(task=task, iter=u, name='forg_tag', group='test', value=100 * forg_tag[task, u])

        #print("Current model evaluates current task and next task")
        if task < (args.tasks-1):
            # Evaluate precision ---  current model on current task
            tmpsiipre = evaluate.validate_one_task(
                copy.deepcopy(net), copy.deepcopy(tgt_te[task]), verbose=False, test_size=None,
                allowed_classes=classes_list[task] if (scenario =="class" or scenario=="task") else None
                )
            siiPrecVec.append(tmpsiipre)
            # Evaluate precision -- current model on next task
            tmpsminusii = evaluate.validate_one_task(
                copy.deepcopy(net), copy.deepcopy(tgt_te[task+1]), verbose=False, test_size=None,
                allowed_classes=classes_list[task] if (scenario =="class" or scenario=="task") else None
            ) 
            sminusiiPrecVec.append(tmpsminusii)

        if args.analysis:
            # t-SNE plots
            if issubclass(Appr,Continous_DA_Appr):
                feature_extractor = deepcopy(net.classifier).to(args.device)
                m_type = 1
            elif issubclass(Appr, dom_adapt_Appr):
                feature_extractor = deepcopy(net.encoder).to(args.device)
                m_type = 2
            elif issubclass(Appr, Inc_Learning_Appr):
                feature_extractor = deepcopy(net.model).to(args.device)
                m_type = 2
            tsne_dir    = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.p_dir[2:])
            tsne_file   = os.path.join(tsne_dir, param_stamp + 'tSNE {} {}2{} {} at Task {}.png'.format((args.approach).upper(), s, t, args.network, task))
            
            # if not os.path.exists(tsne_file):
            from utils import visualize
            visualize(src_te[:(task+1)],tgt_te[:(task+1)],feature_extractor, (task+1),
                        filename=tsne_file, colors=colors[:(task+1)],
                        batch_size=args.batch,device=args.device, m_type=m_type)

            m_dir = os.path.join(os.path.join(os.path.join(args.r_dir,TypeAblation), TypeAblation + '_' + args.approach),'models')
            m_file = os.path.join(m_dir, param_stamp + ' Task {}.pt'.format(task))
            torch.save(feature_extractor.to('cpu'), m_file)

        # Save
        print('Save at ' + os.path.join(r_dir, full_exp_name))
        logger.log_result(acc_taw, name="acc_taw", step=task)
        logger.log_result(acc_tag, name="acc_tag", step=task)
        logger.log_result(forg_taw, name="forg_taw", step=task)
        logger.log_result(forg_tag, name="forg_tag", step=task)
        logger.save_model(net.state_dict(), task=task)
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw", step=task)
        logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), name="avg_accs_tag", step=task)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw", step=task)
        logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name="wavg_accs_tag", step=task)

        # # Last layer analysis
        # if args.last_layer_analysis:
        #     weights, biases = last_layer_analysis(net.heads, task, taskcla, y_lim=True)
        #     logger.log_figure(name='weights', iter=task, figure=weights)
        #     logger.log_figure(name='bias', iter=task, figure=biases)

        #     # Output sorted weights and biases
        #     weights, biases = last_layer_analysis(net.heads, task, taskcla, y_lim=True, sort_weights=True)
        #     logger.log_figure(name='weights', iter=task, figure=weights)
        #     logger.log_figure(name='bias', iter=task, figure=biases)
    
    # Print Summary
    print_summary(acc_taw, acc_tag, forg_taw, forg_tag)

    # Get total training-time in seconds, and write to file
    training_time = time.time() - start

    # time_file = open("{}/time-{}.txt".format(r_dir, param_stamp), 'w')
    # time_file.write('{}\n'.format(training_time))
    # time_file.close()

    print("#"*108)
    if args.analysis:
        from utils import plot_acc
        figname = os.path.join(args.p_dir, param_stamp+'-acc_class plot.jpg')
        plot_acc(appr.acc_task1, args, figname, 'Accuracy')
        figname = os.path.join(args.p_dir, param_stamp+'-acc_domain plot.jpg')
        plot_acc(appr.acc_task1_, args, figname, 'Accuracy')
        figname = os.path.join(args.p_dir, param_stamp+'-ave query acc_class plot.jpg')
        plot_acc(appr.ave_acc, args, figname, 'Average Accuracy')
        figname = os.path.join(args.p_dir, param_stamp+'-ave query acc_domain plot.jpg')
        plot_acc(appr.ave_acc_, args, figname, 'Average Accuracy')
        with open("{}/acc_CIL-{}.pkl".format(r_dir, param_stamp+'.pkl'), 'wb') as f:
            pickle.dump(appr.acc_task1, f)
        with open("{}/acc_DIL-{}.pkl".format(r_dir, param_stamp+'.pkl'), 'wb') as f:
            pickle.dump(appr.acc_task1_, f)
        with open("{}/ave acc_CIL-{}.pkl".format(r_dir, param_stamp+'.pkl'), 'wb') as f:
            pickle.dump(appr.ave_acc, f)
        with open("{}/ave acc_DIL-{}.pkl".format(r_dir, param_stamp+'.pkl'), 'wb') as f:
            pickle.dump(appr.ave_acc_, f)

    #-------------------------------------------------------------------------------------------------#

    #----------------------------#
    #----- FINAL EVALUATION -----#
    #----------------------------#

    # taking transformations and class indices from first train dataset
    if args.scenario=="class":
        if isinstance(classes_per_task,int):
            classes_list = [list(range(classes_per_task*(i+1))) for i in range(args.tasks)] 
        elif isinstance(classes_per_task,list): 
            classes_list = [list(range(sum(classes_per_task[:i+1]))) for i in range(len(classes_per_task))]   
    elif args.scenario=="task":
        if isinstance(classes_per_task,int):
            classes_list = [list(range(classes_per_task*i, classes_per_task*(i+1))) for i in range(args.tasks)] 
        elif isinstance(classes_per_task,list): 
            classes_list = [list(range(sum(classes_per_task[:i]), sum(classes_per_task[:i+1]))) for i in range(args.tasks)] 
    elif args.scenario=="domain" or args.singlehead:
        classes_list=None

    if verbose:
        print("\n\nEVALUATION RESULTS:")

    # Evaluate precision of final model on full test-set
    precs1 = evaluate.validate(
            copy.deepcopy(net), copy.deepcopy(tgt_te), n_task=tasktotalnum, verbose=False, test_size=None,
            allowed_classes_list=classes_list if args.scenario=="task" else None,
        )
    # precs2 = evaluate.validate(
    #         copy.deepcopy(net), copy.deepcopy(tgt_te), n_task=tasktotalnum, verbose=False, test_size=None,
    #         allowed_classes_list=classes_list,
    #     )
    precs = precs1
    average_precs = sum(precs) / args.tasks
    if verbose:
        print("\n Precision on test-set (softmax classification):")
        for i in range(args.tasks):
            print(" - Task {}: {:.6f}".format(i + 1, precs[i]))
        avgAccuracyTiPrec = (sum(precs) / args.tasks).item()
        print('=> Average precision over all {} tasks on target data: {:.4f}\n'.format(args.tasks, average_precs))

    precs_src1 = evaluate.validate(
            copy.deepcopy(net), copy.deepcopy(src_te), n_task=tasktotalnum, verbose=False, test_size=None,
            allowed_classes_list=classes_list if args.scenario=="task" else None,
        )
    # precs_src2 = evaluate.validate(
    #         copy.deepcopy(net), copy.deepcopy(src_te), n_task=tasktotalnum, verbose=False, test_size=None,
    #         allowed_classes_list=classes_list
    #     )
    precs_src=precs_src1
    average_precs_src = sum(precs_src) / args.tasks
    # -print on screen
    if verbose:
        for i in range(args.tasks):
            print(" - Task {}: {:.4f}".format(i + 1, precs_src[i]))
        print('=> Average precision over all {} tasks on source data: {:.4f}\n'.format(args.tasks, average_precs_src))
    
        # print("#"*50)
        # print("scenario: {}".format(args.scenario))
        # [print(" - Task {}: {:.4f}".format(i + 1, precs2[i])) for i in range(args.tasks)]
        # print("Mean acc. ", sum(precs2) / args.tasks)
        # [print(" - Task {}: {:.4f}".format(i + 1, precs_src2[i])) for i in range(args.tasks)]
        # print("Mean acc. ", sum(precs_src2) / args.tasks)

    ## Backward Transfer Prec  And Foward Transfer Prec
    for i in range(args.tasks-1):
        tmpbackwardtransfer = precs[i] - siiPrecVec[i]
        backwardTransferSTiPrecVec.append(tmpbackwardtransfer.item())
        tmpforwardtransfer = sminusiiPrecVec[i] - initialmodeAccuracyPrecVec_tgt[i+1]
        forwardTransferSTiPrecVec.append(tmpforwardtransfer.item())
    
    bwt = sum(backwardTransferSTiPrecVec) / (args.tasks - 1)
    fwt = sum(forwardTransferSTiPrecVec) / (args.tasks - 1)
    metrics = {}
    metrics["FWT (per task)"] = ['NA'] + backwardTransferSTiPrecVec
    metrics["BWT (per task)"] = backwardTransferSTiPrecVec + ['NA']
    metrics['Acc Query (per task)'] = precs
    metrics['Acc Source (per task)'] = precs_src

    print("Average accuracy") 
    print(avgAccuracyTiPrec)
    print("backwardTransfer")
    print(bwt)
    print("forwardTransfer")
    print(fwt)
    if args.approach=="clamp":
        print("Pseudo threshold")
        print(appr.thresholds)
        with open("{}/pseudo-threshold-{}.txt".format(r_dir, param_stamp), 'w') as f:
            for line in appr.thresholds:
                f.write(f"{line}\n")
    #-------------------------------------------------------------------------------------------------#

    #------------------#
    #----- OUTPUT -----#
    #------------------#

    # # Average precision on full test set
    # output_file = open("{}/prec-{}.txt".format(r_dir, param_stamp), 'w')
    # output_file.write('{}\n'.format(average_precs.item()))
    # output_file.close()
    # output_file = open("{}/prec-src-{}.txt".format(r_dir, param_stamp), 'w')
    # output_file.write('{}\n'.format(average_precs_src.item()))
    # output_file.close()
    # # # assess metrics -dict
    # # metrics_file = open("{}/dict-{}".format(args.r_dir, param_stamp), 'w')
    # # metrics_file.write('{}\n'.format({'BWT':bwt, 'FWT':fwt}))
    # # metrics_file.close()
    # # # all metrics
    # # metrics_file = "{}/metrics-{}.npy".format(args.r_dir, param_stamp)
    # # np.save(metrics_file, metrics)
    # # save final model
    # # del model.mem_sampler
    # # del model.trans_mem_sampler
    # # model_file = "{}/model-{}.pt".format(args.m_dir, param_stamp)
    # # torch.save(net, model_file)
    
    print('Done!')
    return precs, precs_src, bwt, fwt, training_time


def run_main(parser, args):


    # print('=' * 108)
    # print('Arguments =')
    # for arg in np.sort(list(vars(args).keys())):
    #     print(arg + ':', getattr(args, arg))
    # print('=' * 108)

    r_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.r_dir[2:])
    args.r_dir = r_dir

    # Use cuda?
    args.device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'

    if args.source == DatasetsType.office31:
        src_list = Office31_src
        tgt_list = Office31_tgt
    elif args.source == DatasetsType.officehome:
        src_list = OfficeHome_src
        tgt_list = OfficeHome_tgt
    elif args.source == DatasetsType.mnist or args.source == DatasetsType.usps:
        src_list = [args.source]
        tgt_list = [args.target]

    # create folder to save performance file
    if 'office' in args.source:
        source_folder = args.source
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        file_path = 'outputs/%s/' % (source_folder)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

    for s,t in zip(src_list, tgt_list):
        TypeAblation  = '{}2{}'.format(s, t)

        if 'office' in args.source:
            file_path = 'outputs/%s/%s' % (source_folder,TypeAblation)
        else:
            file_path = 'outputs/%s' % (TypeAblation)
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            
        print(TypeAblation)
        # args.source = s
        # args.target = t

        from param_stamp import get_param_stamp_from_args
        param_stamp = get_param_stamp_from_args(args)
        result_dir = file_path + '/' + param_stamp \
            + '.pkl'
        print(os.path.abspath(result_dir))
        

        if os.path.isfile(result_dir):
        # if False:

            w1, w2, w3, w4, w5 = pickle.load(open(result_dir, 'rb'), encoding='utf-8')

            # print('-'*108)
            # print('Acc on T: %.4f +/- %.4f' % (np.mean(w1), np.std(w1)))
            # print('Acc on S: %.4f +/- %.4f' % (np.mean(w2), np.std(w2)))
            # print('bwt: %.4f +/- %.4f' % (np.mean(w3), np.std(w3)))
            # print('fwt: %.4f +/- %.4f' % (np.mean(w4), np.std(w4)))
            # print('Training time: %.4f +/- %.4f' % (np.mean(w5), np.std(w5)))

            # print("*"*108)
            # print(args.approach, s, "-->", t)
            # print('-'*108)
            # print("Query Acc \t",end="")
            # [print("Task {} \t\t".format(i), end="") for i in range(len(w1[0]))]
            # print("Ave acc")
            # for i_run in range(args.runs):
            #     print("Round {}\t\t".format(i_run),end="")
            #     [print("{:.6f}%\t".format(w1[i_run][i]*100),end="") for i in range(len(w1[i_run]))]
            #     print("{:.6f}%".format(np.mean(w1[i_run])*100))
            # print('-'*108)
            # print("Source Acc \t",end="")
            # [print("Task {} \t\t".format(i), end="") for i in range(len(w2[i_run]))]
            # print("Ave acc")
            # for i_run in range(args.runs):
            #     print("Round {}\t\t".format(i_run),end="")
            #     [print("{:.6f}%\t".format(w2[i_run][i]*100),end="") for i in range(len(w2[i_run]))]
            #     print("{:.6f}%".format(np.mean(w2[i_run])*100))
            # print('-'*108)
            print('Acc on T: %.4f +/- %.4f' % (np.mean(np.mean(w1,1)), np.std(np.std(w1,1))))
            print('Acc on S: %.4f +/- %.4f' % (np.mean(w2), np.std(np.std(w2,1))))
            print('bwt: %.4f +/- %.4f' % (np.mean(w3), np.std(w3)))
            print('fwt: %.4f +/- %.4f' % (np.mean(w4), np.std(w4)))
            print('Training time: %.4f +/- %.4f' % (np.mean(w5), np.std(w5)))
            # print("*"*108)
            # print('End!\n')
        else:
            summaryacc = [] # added
            sumsourceacc = []
            trtime = []
            FWT = []
            BWT = []
            for iRoundTest in range(args.runs):


                if args.no_cudnn_deterministic:
                    print('WARNING: CUDNN Deterministic will be disabled.')
                    import utils
                    utils.cudnn_deterministic = False
                    
                # # Multiple gpus
                # if torch.cuda.device_count() > 1:
                #     self.C = torch.nn.DataParallel(C)
                #     self.C.to(self.device)

                # create result folder
                if not os.path.exists(args.r_dir+'/'+TypeAblation):
                    os.makedirs(args.r_dir+'/'+TypeAblation)
                cur_path = os.path.dirname(os.path.abspath(__file__))
                folder_i = os.path.join(cur_path, args.p_dir[2:])
                if not os.path.exists(folder_i):
                    os.makedirs(folder_i)
                folder_i = os.path.join(cur_path, "images")
                if not os.path.exists(folder_i):
                    os.makedirs(folder_i)
                # -run experiment
                acc_t, acc_s, bwt, fwt, training_time = run(s,t, parser, args, iRoundTest, param_stamp, verbose=True)

                # dir_ = os.path.join(args.r_dir, TypeAblation)
                # filename = os.path.join(dir_, param_stamp + '-Result{}.pkl'.format(iRoundTest))
                # if not os.path.exists(os.path.dirname(filename)):
                #     try:
                #         os.makedirs(os.path.dirname(filename))
                #     except OSError as exc:  # Guard against race condition
                #         raise
                # with open(filename, 'wb') as f:
                #     pickle.dump([acc_t, acc_s, bwt, fwt, training_time], f)


                print("==============Results, Accummulated Average===============")
                meanResults = np.round_(np.mean(acc_t, 0), decimals=6)
                stdResults = np.round_(np.std(acc_t, 0), decimals=4)
                print("Target accuracy of round {}: {} +/- {}".format(iRoundTest,meanResults*100,stdResults))
                # sumsourceacc.append(np.mean(acc_s))
                # summaryacc.append(np.mean(acc_t))
                sumsourceacc.append([i.item() for i in acc_s])
                summaryacc.append([i.item() for i in acc_t])
                trtime.append(training_time)
                FWT.append(fwt)
                BWT.append(bwt)
            print('========== ' + str(args.runs) + 'times results' + TypeAblation+' ==========')
            print('%.4f +/- %.4f' % (np.mean(summaryacc),np.std(summaryacc)))
            print('%.4f +/- %.4f' % (np.mean(sumsourceacc),np.std(sumsourceacc)))
            print('%.2f +/- %.2f' % (np.mean(trtime),np.std(trtime)))

            try:
                with open(result_dir, 'wb') as f:
                    pickle.dump([summaryacc, sumsourceacc, BWT, FWT, trtime], f)
            except Exception as e:
                print('Save failed:{}'.format(e))