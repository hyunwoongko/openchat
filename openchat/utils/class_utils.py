import logging
from parlai.core.agents import NOCOPY_ARGS, compare_init_model_opts
from parlai.core.opt import Opt
from parlai.utils.io import PathManager
from parlai.utils.misc import warn_once


def inherit(obj, superclasses):
    obj.__class__ = type(obj.__class__.__name__, superclasses, dict())


def create_agent_from_opt_file_and_model_class(opt, model_class):
    model_file = opt['model_file']
    optfile = model_file + '.opt'

    if not PathManager.exists(optfile):
        return None

    opt_from_file = Opt.load(optfile)

    # delete args that we do not want to copy over when loading the model
    for arg in NOCOPY_ARGS:
        if arg in opt_from_file:
            del opt_from_file[arg]

    # only override opts specified in 'override' dict
    if opt.get('override'):
        for k, v in opt['override'].items():
            if k in opt_from_file and str(v) != str(opt_from_file.get(k)):
                logging.warn(
                    f'Overriding opt["{k}"] to {v} (previously: {opt_from_file.get(k)})'
                )
            opt_from_file[k] = v

    if hasattr(model_class, 'upgrade_opt'):
        opt_from_file = model_class.upgrade_opt(opt_from_file)

    # add model arguments to opt_from_file if they aren't in opt_from_file already
    for k, v in opt.items():
        if k not in opt_from_file:
            opt_from_file[k] = v

    # update model file path to the one set by opt
    opt_from_file['model_file'] = model_file
    # update init model path to the one set by opt
    # NOTE: this step is necessary when for example the 'init_model' is
    # set by the Train Loop (as is the case when loading from checkpoint)
    if opt.get('init_model') is not None:
        opt_from_file['init_model'] = opt['init_model']

    # update dict file path
    if not opt_from_file.get('dict_file'):
        old_dict_file = None
        opt_from_file['dict_file'] = model_file + '.dict'
    elif opt_from_file.get('dict_file') and not PathManager.exists(
            opt_from_file['dict_file']):
        old_dict_file = opt_from_file['dict_file']
        opt_from_file['dict_file'] = model_file + '.dict'
    if not PathManager.exists(opt_from_file['dict_file']):
        warn_once(
            'WARNING: Neither the specified dict file ({}) nor the '
            '`model_file`.dict file ({}) exists, check to make sure either '
            'is correct. This may manifest as a shape mismatch later '
            'on.'.format(old_dict_file, opt_from_file['dict_file']))

    # if we want to load weights from --init-model, compare opts with
    # loaded ones
    compare_init_model_opts(opt, opt_from_file)
    return model_class(opt_from_file)
