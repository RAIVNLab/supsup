import math
import torch
import torch.nn as nn

from args import args

def signed_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, args.mode)
    gain = nn.init.calculate_gain(args.nonlinearity)
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std

def unsigned_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, args.mode)
    gain = nn.init.calculate_gain(args.nonlinearity)
    std = gain / math.sqrt(fan)
    module.weight.data = torch.ones_like(module.weight.data) * std

def kaiming_normal(module):
    nn.init.kaiming_normal_(
        module.weight, mode=args.mode, nonlinearity=args.nonlinearity
    )

def kaiming_uniform(module):
    nn.init.kaiming_uniform_(
        module.weight, mode=args.mode, nonlinearity=args.nonlinearity
    )

def xavier_normal(module):
    nn.init.xavier_normal_(
        module.weight
    )

def glorot_uniform(module):
    nn.init.xavier_uniform_(
        module.weight
    )

def xavier_constant(module):
    fan = nn.init._calculate_correct_fan(module.weight, args.mode)
    gain = 1.0
    std = gain / math.sqrt(fan)
    module.weight.data = module.weight.data.sign() * std

def default(module):
    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
