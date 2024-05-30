import argparse
import random
import torch.backends.cudnn as cudnn
from train_PudNet import *
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cudnn.benchmark = True
cudnn.fastest = True

FLAG_PLATFORM = 'colab'

parser = argparse.ArgumentParser(description='Train the unet network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

if FLAG_PLATFORM == 'colab':
    parser.add_argument('--gpu_ids', default='2', dest='gpu_ids')

    parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')  
    parser.add_argument('--dir_log', default='./log', dest='dir_log')  
    parser.add_argument('--dir_result', default='./results', dest='dir_result') 
    parser.add_argument('--dir_data', default='../datasets', dest='dir_data')  
elif FLAG_PLATFORM == 'laptop':
    parser.add_argument('--gpu_ids', default='-1', dest='gpu_ids')

    parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
    parser.add_argument('--dir_log', default='./log', dest='dir_log')
    parser.add_argument('--dir_result', default='./results', dest='dir_result')
    parser.add_argument('--dir_data', default='./datasets', dest='dir_data')

parser.add_argument('--seed', default=100, type=int, help='random seed')
parser.add_argument('--num_subdata', default=20000, type=int, help='the number of the subdatasets')  
parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode') 
parser.add_argument('--train_continue', default='on', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--scope', default='PudNet_uniAHRU', dest='scope')  
parser.add_argument('--norm', type=str, default='bnorm', dest='norm')

parser.add_argument('--name_data', type=str, default='imagenet', dest='name_data')  

parser.add_argument('--num_epoch', type=int,  default=100, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=50, dest='batch_size')  

parser.add_argument('--lr_G', type=float, default=1e-3, dest='lr_G')

parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop'], dest='optim')
parser.add_argument('--beta1', default=0.5, dest='beta1')  

parser.add_argument('--ny_in', type=int, default=128, dest='ny_in')  
parser.add_argument('--nx_in', type=int, default=128, dest='nx_in')  
parser.add_argument('--nch_in', type=int, default=3, dest='nch_in')  

parser.add_argument('--ny_load', type=int, default=256, dest='ny_load')
parser.add_argument('--nx_load', type=int, default=256, dest='nx_load')
parser.add_argument('--nch_load', type=int, default=3, dest='nch_load')

parser.add_argument('--ny_out', type=int, default=256, dest='ny_out')
parser.add_argument('--nx_out', type=int, default=256, dest='nx_out')
parser.add_argument('--nch_out', type=int, default=3, dest='nch_out')

parser.add_argument('--nch_ker', type=int, default=64, dest='nch_ker')

parser.add_argument('--data_type', default='float32', dest='data_type')  

parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers') 

parser.add_argument('--num_freq_disp', type=int,  default=50, dest='num_freq_disp')  

parser.add_argument('--hid_hyper', default=8, type=int, help='the hidden dimension in HyperNetwork.') 
parser.add_argument('--hid_dim', default=128, type=int, help='the dec_hid_dim in GRU.')
parser.add_argument('--gru_layers', default=2, type=int, help='the dec_hid_dim in GRU.')
parser.add_argument('--out_dim', default=128, type=int, help='the output_dim in GRU.')
parser.add_argument('--gru_drop', default=0., type=float, help='the dropout rate')
parser.add_argument('--dec_drop', default=0., type=float, help='the dropout rate')
parser.add_argument('--skt_a', default=0.1, type=float, help='the weight of sketch for concate')
parser.add_argument('--dropout_rate', default=0., type=float, help='the dropout rate around linear layer.')
parser.add_argument('--sgm', default=25, type=int, help='adding zero mean Gaussian noise with standard deviation.')


PARSER = Parser(parser)

def main():
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    torch.cuda.manual_seed_all(ARGS.seed)  
    torch.cuda.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed) 
    random.seed(ARGS.seed) 
    torch.manual_seed(ARGS.seed)
    cudnn.deterministic = True

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        TRAINER.train(ARGS)
    elif ARGS.mode == 'test':
        TRAINER.test(ARGS)

if __name__ == '__main__':
    main()