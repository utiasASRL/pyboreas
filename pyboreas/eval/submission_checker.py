import os
import os.path as osp
from pathlib import Path
import argparse
import zipfile
import yaml
import shutil
from pyboreas.data.splits import odom_test

def check_yaml(yml):
    keys = ['benchmark', 'methodname', '2d', 'author', 'papertitle', 'paperurl',
        'venue', 'year', 'runtimeseconds', 'computer']
    for key in keys:
        try:
            yml[key]
        except KeyError:
        	print('missing key')
            return False
    if yml['benchmark'] != 'odometry' and \
        yml['benchmark'] != 'localization' and \
        yml['benchmark'] != 'detection':
        print('benchmark incorrect')
        return False

    if not isinstance(yml['2d'], bool):
    	print('2d must be bool')
    	return False

    if len(yml['methodname']) > 30 or yml['methodname'] is None:
    	print('bad methodname')
    	return False

    # check length of metadata
    if len(yml['author']) > 100 or \
    	len(yml['papertitle']) > 150 or \
    	len(yml['paperurl']) > 500 or \
    	len(yml['venue']) > 10 or \
    	len(str(yml['year'])) > 4 or \
    	len(str(yml['runtimeseconds'])) > 10 or \
    	len(yml['computer']) > 50:
    	print('metadata too long')
    	return False

    return True

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--file', default='test-odometry.zip')
	parser.add_argument('--test_times', default='detection_test_times.txt')
	args = parser.parse_args()

	benchmarks = ['odometry', 'localization', 'detection']

	bench = Path(args.file).stem.split('-')[-1].split('.')[0]
	if bench not in benchmarks:
		raise Exception('{} not one of : {}'.format(bench, benchmarks))

	tmp = 'tmp_results'
	if not osp.exists(tmp):
		os.mkdir(tmp)
	else:
		shutil.rmtree(tmp)
		os.mkdir(tmp)
	with zipfile.ZipFile(args.file, 'r') as zip_ref:
		zip_ref.extractall(tmp)

	files = sorted(os.listdir(tmp))
	if 'metadata.yaml' not in files:
		raise Exception('metadata.yaml not found')
	files.remove('metadata.yaml')

	with open(osp.join(tmp, 'metadata.yaml')) as f:
		meta = yaml.safe_load(f)

	if not check_yaml(meta):
		raise Exception('metadata.yaml not correctly formatted')

	if bench != meta['benchmark']:
		raise Exception('{} != {}'.format(bench, meta['benchmark']))

	if bench == 'odometry' or bench == 'localization':
		gt_seqs = [x[0] for x in odom_test]
		pred_seqs = [f.split('.')[0] for f in files]
		if len(gt_seqs) != len(pred_seqs):
			raise Exception('number of predictions does not match number of gt seqs')
		for pred in pred_seqs:
			if pred not in gt_seqs:
				raise Exception('pred: {} does not match gt seqs'.format(pred))
	elif bench == 'detection':
		if not osp.exists(args.test_times):
			raise Exception('{} not found'.format(args.test_times))
		with open(args.test_times) as f:
			gt_times = f.readlines()
		preds = sorted([f.split('.')[0] for f in files])
		if len(preds) != len(gt_times):
			raise Exception('number of predictions does not match number of gt times')
		for pred in preds:
			if pred not in gt_times:
				raise Exception('pred: {} does not match a gt time'.format(pred))

	shutil.rmtree(tmp)
	print('Submission checker PASSED')
