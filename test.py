from transformers import HfArgumentParser
from dataclasses import dataclass, field

@dataclass
class A:
    a: str = field()

@dataclass 
class B:
    b: str = field(
        default='test'
    )
    
parser = HfArgumentParser(
    (
        A,
        B
    )
)



from arguments import get_args

model_args, data_args, training_args = get_args()

print('model_args: ', model_args)
print(training_args.do_train)