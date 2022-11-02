
from model.model.seg import seg
from model.model.mass_detection import mass_detection
from model.model.lnet import lnet

from model.model.lm_net import lm_net
from model.model.lmc_net import lmc_net

from model.model.lm_vnet import lm_vnet
from model.model.lmc_vnet import lmc_vnet





model_list = {
    "seg": seg,
    "mass_detection": mass_detection,
    "lnet": lnet,
    "lm_net":lm_net,
    "lmc_net": lmc_net,
    "lm_vnet": lm_vnet,
    "lmc_vnet": lmc_vnet,

}

def get_model(name, config):
    print(name)
    if name in model_list.keys():        
        return model_list[name](config)
    else:
        raise NotImplementedError
