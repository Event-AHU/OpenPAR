from config import argument_parser
from tools.training import run_training
if __name__ == '__main__':
    run_training(argument_parser(), stage='stage1')
