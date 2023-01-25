def get_param_stamp_from_args(args):
    '''To get param-stamp a bit quicker.'''

    appr = args.approach
    scenario = args.scenario
    model_name = args.network
    
    # Args -- Network
    if args.approach !=  'clamp':
        if args.num_exemplars_per_class1 == 0:
            param_stamp = "{}-{} incre-{}-lr {}-bs {}-e {}-pe {}-fc {}-w/o batchnorm-{}runs".format(
                appr,
                scenario,
                model_name,
                args.lr1,
                args.batch,
                args.batch_d,
                args.epoch,
                args.warmup_nepochs,
                args.fc_units,
                args.runs
            )
        else:
            param_stamp = "{}-{} incre-{}-lr {}-bs {}-domain bs {}-e {}-pe {}-fc {}-(b{})-{}runs".format(
                appr,
                scenario,
                model_name,
                args.lr1,
                args.batch,
                args.batch_d,
                args.epoch,
                args.warmup_nepochs,
                args.fc_units,
                args.num_exemplars_per_class1,
                args.runs
            )
    elif args.approach == 'clamp':
        if (not args.meta) and (not args.pseudo) and (not args.domain):
            ablation_stamp = 'plain'
        else:
            ablation_stamp = 'pseudo' if args.pseudo else ''
            ablation_stamp = ablation_stamp+'-meta' if args.meta else ablation_stamp
            ablation_stamp = ablation_stamp+'-domain' if args.domain else ablation_stamp

        param_stamp = "{}-{} incre-{}-lr{}-lr asse{}-bs{}-domain bs{}-e{}-pe{}-ie{}-oe{}-fc{}-alpha{}-b({}n{})-{}-{}runs".format(
            appr,
            scenario,
            model_name,
            args.lr1,
            args.lr2,
            args.batch,
            args.batch_d,
            args.epoch,
            args.warmup_nepochs,
            args.epochIn,
            args.epochOut,
            args.fc_units,
            args.alpha,
            args.num_exemplars_per_class1,
            args.num_exemplars_per_class2,
            ablation_stamp,
            args.runs
        )

    return param_stamp



def get_param_stamp(args, model_name, verbose=True, replay=False, replay_model_name=None):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for task
    multi_n_stamp = "{n}-{set}".format(n=args.tasks, set=args.scenario) if hasattr(args, "tasks") else ""
    task_stamp = "{exp}{multi_n}".format(exp=args.experiment, multi_n=multi_n_stamp)
    if verbose:
        print(" --> task:          "+task_stamp)

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:         "+model_stamp)

    # -for hyper-parameters
    hyper_stamp = "{i_e}{num}-lr{lr}{lrg}-b{bsz}-{optim}".format(
        i_e="e" if args.iters is None else "i", num=args.epochs if args.iters is None else args.iters, lr=args.lr,
        lrg=("" if args.lr==args.lr_gen else "-lrG{}".format(args.lr_gen)) if hasattr(args, "lr_gen") else "",
        bsz=args.batch, optim=args.optimizer,
    )
    if verbose:
        print(" --> hyper-params:  " + hyper_stamp)

    # -for EWC / SI
    if hasattr(args, 'ewc') and ((args.ewc_lambda>0 and args.ewc) or (args.si_c>0 and args.si)):
        ewc_stamp = "EWC{l}-{fi}{o}".format(
            l=args.ewc_lambda,
            fi="{}{}".format("N" if args.fisher_n is None else args.fisher_n, "E" if args.emp_fi else ""),
            o="-O{}".format(args.gamma) if args.online else "",
        ) if (args.ewc_lambda>0 and args.ewc) else ""
        si_stamp = "SI{c}-{eps}".format(c=args.si_c, eps=args.epsilon) if (args.si_c>0 and args.si) else ""
        both = "--" if (args.ewc_lambda>0 and args.ewc) and (args.si_c>0 and args.si) else ""
        if verbose and args.ewc_lambda>0 and args.ewc:
            print(" --> EWC:           " + ewc_stamp)
        if verbose and args.si_c>0 and args.si:
            print(" --> SI:            " + si_stamp)
    ewc_stamp = "--{}{}{}".format(ewc_stamp, both, si_stamp) if (
        hasattr(args, 'ewc') and ((args.ewc_lambda>0 and args.ewc) or (args.si_c>0 and args.si))
    ) else ""

    # -for XdG
    xdg_stamp = ""
    if (hasattr(args, 'xdg') and args.xdg) and (hasattr(args, "gating_prop") and args.gating_prop>0):
        xdg_stamp = "--XdG{}".format(args.gating_prop)
        if verbose:
            print(" --> XdG:           " + "gating = {}".format(args.gating_prop))

    # -for replay
    if replay:
        replay_stamp = "{rep}{KD}{agem}{model}{gi}".format(
            rep=args.replay,
            KD="-KD{}".format(args.temp) if args.distill else "",
            agem="-aGEM" if args.agem else "",
            model="" if (replay_model_name is None) else "-{}".format(replay_model_name),
            gi="-gi{}".format(args.gen_iters) if (
                hasattr(args, "gen_iters") and (replay_model_name is not None) and (not args.iters==args.gen_iters)
            ) else ""
        )
        if verbose:
            print(" --> replay:        " + replay_stamp)
    replay_stamp = "--{}".format(replay_stamp) if replay else ""

    # -for exemplars / iCaRL
    exemplar_stamp = ""
    if hasattr(args, 'use_exemplars') and (args.add_exemplars or args.use_exemplars or args.replay=="exemplars"):
        exemplar_opts = "b{}{}{}".format(args.budget, "H" if args.herding else "", "N" if args.norm_exemplars else "")
        use = "{}{}".format("addEx-" if args.add_exemplars else "", "useEx-" if args.use_exemplars else "")
        exemplar_stamp = "--{}{}".format(use, exemplar_opts)
        if verbose:
            print(" --> exemplars:     " + "{}{}".format(use, exemplar_opts))

    # -for binary classification loss
    binLoss_stamp = ""
    if hasattr(args, 'bce') and args.bce:
        binLoss_stamp = '--BCE_dist' if (args.bce_distill and args.scenario=="class") else '--BCE'

    # --> combine
    param_stamp = "{}--{}--{}{}{}{}{}{}{}".format(
        task_stamp, model_stamp, hyper_stamp, ewc_stamp, xdg_stamp, replay_stamp, exemplar_stamp, binLoss_stamp,
        "-s{}".format(args.seed) if not args.seed==0 else "",
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp