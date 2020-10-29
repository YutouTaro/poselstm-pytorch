
def create_model(opt):
    model = None
    print("Network model: " + opt.model)
    if opt.model == 'posenet':
        from .posenet_model import PoseNetModel
        model = PoseNetModel()
    elif opt.model == 'poselstm':
        from .poselstm_model import PoseLSTModel
        model = PoseLSTModel()
    elif opt.model == 'fcnlstm':
        from .fcnlstm_model import FCNLSTModel
        model = FCNLSTModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
