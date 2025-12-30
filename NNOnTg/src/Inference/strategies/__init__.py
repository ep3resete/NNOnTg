from .top_p import TopPStrategy
from .top_k import TopKStrategy
from .greedy import GreedyStrategy
from .sampling import SamplingStrategy

STRATEGIES = {
    'greedy': GreedyStrategy,
    'sample': SamplingStrategy,
    'top_k': TopKStrategy,
    'top_p': TopPStrategy
}