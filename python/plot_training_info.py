import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import numpy as np

import os, sys, argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Plot Training Loss')
	parser.add_argument('infofile', nargs=1, help='training info file', type=str)
	parser.add_argument('-save', help='save figure', action='store_true')
	parser.add_argument('-live', help='live update', action='store_true')
	
	if len(sys.argv) == 1:
		parser.print_help()
		sys.exit(1)

	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = parse_args()

	if (args.save):
		fname = args.infofile[0]+'.png'
	else:
		if (args.live):
			plt.ion()
			plt.show()

	while True:
		with open(args.infofile[0], 'rb') as f:
			N = np.fromfile(f, np.int32, 1);
			batch_size = np.fromfile(f, np.int32, 1);
			subdiv = np.fromfile(f, np.int32, 1);
			siter = np.fromfile(f, np.int32, 1)
			n_loss = np.fromfile(f, np.int32, 1)
			sub_loss_id = np.fromfile(f, np.int32, n_loss[0])
			losses = np.fromfile(f, np.float32, -1)

		if (losses.shape[0] % n_loss[0] != 0):
			cutoff = losses.shape[0] % n_loss[0]
			losses = losses[0:losses.shape[0]-cutoff]
		losses = np.transpose(np.reshape(losses,(-1,n_loss[0])))

		miter = siter + losses.shape[1]
		t = np.arange(siter,miter,1)
		t = t.astype(float) * batch_size.astype(float) * subdiv.astype(float) / N.astype(float)
		plt.clf()

		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.grid(color='k', linestyle='-', linewidth=0.5, alpha=0.2)

		for i in range(n_loss[0]):
			#loss
			graph_label = 'Layer ' + str(sub_loss_id[i])
			graph_color = 'C' + str(i%10)
			plt.plot(t, losses[i], color=graph_color, label=graph_label)
		plt.legend(loc = 0)

		if (args.save):
			print('Save figure to ' + fname)
			plt.savefig(fname, dpi='figure')
			break
		else:
			plt.draw()
			if (args.live):
				plt.pause(0.01)
			else:
				plt.show()

		if (not args.live):
			break

		if (not plt.get_fignums()):
			break
