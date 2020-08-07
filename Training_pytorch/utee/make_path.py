import os


def makepath(args,except_list):
    path_list = []
    for k in args.__dict__:
        if k not in except_list:
            path_list.append(k)
    path_list.sort()
    for level in path_list:
        args.logdir = os.path.join(args.logdir,level+"="+str(args.__dict__[level]))
    print(args.logdir)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    return args

def makefile(args,except_list):
    path_list = []
    for k in args.__dict__:
        if k not in except_list:
            path_list.append(k)
    path_list.sort()
    log_name = ''
    for level in path_list:
        log_name = log_name+'_'+str(level)+"="+str(args.__dict__[level])
    print(log_name)
    return log_name
