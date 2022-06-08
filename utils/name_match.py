from avalanche.OnlineContinualLearning.agents.gdumb import Gdumb
from avalanche.OnlineContinualLearning.continuum.dataset_scripts.cifar100 import CIFAR100
from avalanche.OnlineContinualLearning.continuum.dataset_scripts.cifar10 import CIFAR10
from avalanche.OnlineContinualLearning.continuum.dataset_scripts.core50 import CORE50
from avalanche.OnlineContinualLearning.continuum.dataset_scripts.mini_imagenet import Mini_ImageNet
from avalanche.OnlineContinualLearning.continuum.dataset_scripts.openloris import OpenLORIS
from avalanche.OnlineContinualLearning.agents.exp_replay import ExperienceReplay
# from avalanche.OnlineContinualLearning.agents.agem import AGEM
# from avalanche.OnlineContinualLearning.agents.ewc_pp import EWC_pp
# from avalanche.OnlineContinualLearning.agents.cndpm import Cndpm
# from avalanche.OnlineContinualLearning.agents.lwf import Lwf
# from avalanche.OnlineContinualLearning.agents.icarl import Icarl
# from avalanche.OnlineContinualLearning.agents.scr import SupContrastReplay
from avalanche.OnlineContinualLearning.utils.buffer.random_retrieve import Random_retrieve
from avalanche.OnlineContinualLearning.utils.buffer.reservoir_update import Reservoir_update
from avalanche.OnlineContinualLearning.utils.buffer.mir_retrieve import MIR_retrieve
# from avalanche.OnlineContinualLearning.utils.buffer.gss_greedy_update import GSSGreedyUpdate
# from avalanche.OnlineContinualLearning.utils.buffer.aser_retrieve import ASER_retrieve
# from avalanche.OnlineContinualLearning.utils.buffer.aser_update import ASER_update
# from avalanche.OnlineContinualLearning.utils.buffer.sc_retrieve import Match_retrieve
# from avalanche.OnlineContinualLearning.utils.buffer.mem_match import MemMatch_retrieve

data_objects = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10,
    'core50': CORE50,
    'mini_imagenet': Mini_ImageNet,
    'openloris': OpenLORIS
}

agents = {
    'ER': ExperienceReplay,
    # 'EWC': EWC_pp,
    # 'AGEM': AGEM,
    # 'CNDPM': Cndpm,
    # 'LWF': Lwf,
    # 'ICARL': Icarl,
    # 'GDUMB': Gdumb,
    # 'SCR': SupContrastReplay,
}

retrieve_methods = {
    'MIR': MIR_retrieve,
    'random': Random_retrieve,
    # 'ASER': ASER_retrieve,
    # 'match': Match_retrieve,
    # 'mem_match': MemMatch_retrieve

}

update_methods = {
    'random': Reservoir_update,
    # 'GSS': GSSGreedyUpdate,
    # 'ASER': ASER_update
}

