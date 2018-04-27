from trainer import Trainer
import argparse
from PIL import Image
import os
from build_vocab import *

parser = argparse.ArgumentParser()
parser.add_argument("--type", default='gan')
parser.add_argument("--lr", default=0.0002, type=float)
parser.add_argument("--l1_coef", default=50, type=float)
parser.add_argument("--l2_coef", default=100, type=float)
parser.add_argument("--diter", default=5, type=int)
parser.add_argument("--cls", default=False, action='store_true')
parser.add_argument("--interp", default=False, action='store_true')
parser.add_argument("--vis_screen", default='gan')
parser.add_argument("--save_path", default='')
parser.add_argument("--inference", default=False, action='store_true')
parser.add_argument('--pre_trained_disc_A', default=None)
parser.add_argument('--pre_trained_gen_A', default=None)
parser.add_argument('--pre_trained_disc_B', default=None)
parser.add_argument('--pre_trained_gen_B', default=None)
parser.add_argument('--dataset', default='flowers')
parser.add_argument('--split', default=0, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--epochs', default=55, type=int)
args = parser.parse_args()

trainer = Trainer(type=args.type,
dataset=args.dataset,
split=args.split,
lr=args.lr,
diter=args.diter,
vis_screen=args.vis_screen,
save_path=args.save_path,
l1_coef=args.l1_coef,
l2_coef=args.l2_coef,
pre_trained_disc=args.pre_trained_disc_A,
pre_trained_gen=args.pre_trained_gen_A,
batch_size=args.batch_size,
num_workers=args.num_workers,
epochs=args.epochs,
pre_trained_disc_B=args.pre_trained_disc_B,
pre_trained_gen_B=args.pre_trained_gen_B
)

print(args.inference)
print(args.type)

if not args.inference and args.type!='cycle_gan':
    print("gan")
    trainer.train(args.cls, args.interp)
elif not args.inference and args.type=='cycle_gan':
    print('cycle gan')
    cycle_trainer.train(args.cls)
elif args.inference and args.type=='cycle_gan':
    # trainer.predict()
    print('cycle gan prediction')
    cycle_trainer.predict()
elif args.inference and args.type=='gan':
    print('gan prediction')
    trainer.predict()
elif args.inference and args.type=='stackgan':
    trainer.predict(args.type)
else:
    print('wrong input...')

