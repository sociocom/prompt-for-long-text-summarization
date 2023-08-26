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

args = parser.parse_args_into_dataclasses()
print(args)