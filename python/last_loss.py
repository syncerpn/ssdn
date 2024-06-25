import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np

import os, sys, argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Plot Training Loss')
	parser.add_argument('infofile', nargs=1, help='training info file', type=str)
	
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	with open(args.infofile[0], 'rb') as f:
		N = np.fromfile(f, np.int32, 1);
		batch_size = np.fromfile(f, np.int32, 1);
		subdiv = np.fromfile(f, np.int32, 1);
		siter = np.fromfile(f, np.int32, 1)
		n_loss = np.fromfile(f, np.int32, 1)
		sub_loss_id = np.fromfile(f, np.int32, n_loss[0])
		losses = np.fromfile(f, np.float32, -1)
		losses = np.transpose(np.reshape(losses,(-1,n_loss[0])))

	miter = siter + losses.shape[1]
	t = np.arange(siter,miter,1)
	t = t.astype(float) * batch_size.astype(float) * subdiv.astype(float) / N.astype(float)

	for i in range(n_loss[0]):
		last_loss = 'Layer ' + str(sub_loss_id[i]) + ' : ' + str(losses[i][-1])
		print(last_loss)