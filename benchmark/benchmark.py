




import numpy
import time
import datetime
import itertools
import subprocess
import os
import fnmatch
from statistics import mean
import csv
import collections
import copy


from argument import *

from utils import *

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'


import random
#Usefull ?
random.seed(10)




class Report(object):
    """
    This class represent a report for a benchmark
    """

    def __init__(self, benchmark):
        """
        This constructor create a new report for a benchmark

        :param benchmark: The benchmark from which a report will be made
        """

        self.res_psnr = []
        self.res_std = []

        self.benchmark = benchmark


    def make_report(self, num_config=0):
        """
        This method take the results made by the tests of different configurations and register them
        """

        input_dir = self.benchmark.params['input_dir'] + "/Test/"

        for root, dirs, files in os.walk(input_dir):
            for file in fnmatch.filter(files, "*.res"):
                print(file)
                with open(os.path.join(root, file), "r") as f:
                    lines = f.read()

                    self.res_psnr.append(float(list(filter(lambda string : string.startswith('psnr'), lines.split('\n')))[0].split(':')[1]))
                    self.res_std.append(float(list(filter(lambda string : string.startswith('std'), lines.split('\n')))[0].split(':')[1]))



    def toCSV(self, num_config=0):
        """
        This method write into the csv file the results made during the benchmark
        """

        #Number of row for the final list to print in the CSV file
        num_row = 100

        #Number of line before the results
        offset = 0

        if(len(self.res_psnr) == 0):
            self.make_report()

        mylist = [[]]
        mydict = collections.defaultdict(lambda: collections.defaultdict(dict))

        with open(self.benchmark.file_path, 'r', newline='') as csvfile:

            reader = csv.reader(csvfile, delimiter=';')
            mylist = list(reader)

        for i in range(len(mylist)):
            if(mylist[i][0] == "Results"):
                offset = i


        mydict[0][1+num_config] = self.benchmark.params['input_dir']

        mydict[2+offset][1+num_config] = self.benchmark.params['num_epochs']
        mydict[3+offset][1+num_config] = self.benchmark.params['num_epochs'] #Not able to take the best performing epoch for now (parse the traing.txt file after using the --perform_validation arg for the training process)
        mydict[4+offset][1+num_config] = mean(self.res_std)
        mydict[5+offset][1+num_config] = mean(self.res_psnr)

        for i in range(len(self.res_std)):
            mydict[6+offset+i][1+num_config] = self.res_std[i]
            mydict[6+offset+i+len(self.res_std)][1+num_config] = self.res_psnr[i]



        final_list = [[''] * (self.benchmark.getNbTest() + 1)] * num_row

        for i in range(len(mylist)):
            final_list[i] = mylist[i]


        for k1 in mydict.keys():
            for k2 in mydict[k1].keys():
                final_list[k1][k2] = copy.deepcopy(mydict[k1][k2])



        with open(self.benchmark.file_path, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)


            spamwriter.writerows(final_list)



class Benchmark(object):
    """
    This class is a tool for making benchmark
    """


    def __init__(self, file_path, sbatch=False):
        """
        This constructor create a new benchmark for a csv file

        :param file_path: The path to the csv file
        """

        self.input_dir = 'Benchmark/benchmark_{}'.format(datetime.datetime.now().strftime("%d_%m_%Y-%H:%M:%S"))

        self.valid_args = vars(parse())

        self.nb_config = 0


        self.file_path = file_path
        self.params = dict()

        self.sbatch = sbatch

        self.getParam()


    def getParam(self, num_config=0):
        """
        This method get the changing param from the csv file
        """

        # Get valid args
        for arg in self.valid_args:
            if(isinstance(self.valid_args[arg], tuple)):
                self.params[arg] = ' '.join([str(i) for i in self.valid_args[arg]])
            else:
                self.params[arg] = self.valid_args[arg]



        # Get modified args
        with open(self.file_path, 'r', newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')

            for row in spamreader:
                if(len(row) - 1 > self.nb_config):
                    self.nb_config = len(row) - 1

                if(len(row) > self.getNbTest() and row[0] in self.valid_args and row[1+num_config] != ''):
                    self.params[row[0]] = row[1+num_config]


        self.params['input_dir'] = self.input_dir + "/Test_{}/".format(num_config)
        self.params['save_test_dir'] = '.'




    def get_sbatch_config(self):

        ret = """

        #!/bin/bash

        #SBATCH -p gpu
        #SBATCH --gres gpu:1

        """

        return ret


    def getNbTest(self):
        """
        This method return the number of test launched by the benchmark
        (Just one configuration at a time for now)
        """
        return self.nb_config

    def getInputDir(self):
        """
        This method return the benchmark working directory
        """
        return '{}'.format(self.input_dir)


    def toString(self):
        """
        This method return a representation of the benchmark
        """
        return """
        Benchmark :
        {}
        """.format(self.params)

    def get_params_string(self):
        """
        This method join in a string all params from the csv file
        """
        temp = ' '.join(["--" + k + " " + str(v) if v != None and v != '' and v != False else "" for k, v in self.params.items()])
        return temp


    def launch_benchmark_data(self):
        """
        This method launch the creation of data's configuration for the different tests (just one for the moment)
        """

        process = []

        list_params = self.get_params_string()

        cmd = '''
        python3 generate_patches_holo_fromMAT.py {} &
        '''.format(list_params).replace("\n", "")


        p = subprocess.Popen(cmd, shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)


        process.append(p)

        p.communicate()

        exit_codes = [p.communicate() for p in process]


        sess_name = extract_sess_name(tuple(self.params['train_patterns'].split(" ")), self.params['train_noises'], self.params['phase_type'], self.params['stride'], self.params['patch_size'], self.params['patch_per_image']).replace(' ','')

        self.params['clean_train'] = os.path.join(self.params['save_dir'], "img_clean_train_" + sess_name + ".npy")
        self.params['noisy_train'] = os.path.join(self.params['save_dir'], "img_noisy_train_" + sess_name + ".npy")


        sess_name = extract_sess_name(tuple(self.params['eval_patterns'].split(" ")), self.params['eval_noises'], self.params['phase_type'], self.params['stride'], self.params['patch_size'], self.params['patch_per_image']).replace(' ','')

        self.params['clean_eval'] = os.path.join(self.params['save_dir'], "img_clean_train_" + sess_name + ".npy")
        self.params['noisy_eval'] = os.path.join(self.params['save_dir'], "img_noisy_train_" + sess_name + ".npy")





    def launch_benchmark_training(self):
        """
        This method launch the training for the different configurations (just one for the moment)
        """

        process = []


        output_dir = '/'
        list_params = self.get_params_string()

        cmd_python = '''
        python3 main_holo.py --output_dir {} {} &
        '''.format(output_dir, list_params).replace("\n", "")

        cmd_path = self.params['input_dir'] + "/cmd_train"


        with open(cmd_path , 'w') as cmd_file:
            if(self.sbatch):
                print(self.get_sbatch_config(), file=cmd_file)

            print(cmd_python, file=cmd_file)


        cmd_bash = '''
        chmod 755 {};
        ./{} &
        '''.format(cmd_path, cmd_path).replace("\n", "")

        if(self.sbatch):
            cmd_bash = '''
            chmod 755 {};
            sbatch {}
            '''.format(cmd_path, cmd_path).replace("\n", "")


        print("\n\n\n")
        print("Training CMD : ", cmd_python)
        print("\n\n\n")


        p = subprocess.Popen(cmd_bash, shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)


        process.append(p)

        p.communicate()

        exit_codes = [p.communicate() for p in process]


    def launch_benchmark_testing(self):
        """
        This method launch the tests for the different configurations (just one for the moment)
        """

        process = []


        save_test_dir = '.'
        list_params = self.get_params_string()



        cmd_python = '''
        python3 main_holo.py --test_mode --save_test_dir {} {} &
        '''.format(save_test_dir, list_params).replace("\n", "")


        cmd_path = self.params['input_dir'] + "/cmd_test"


        with open(cmd_path , 'w') as cmd_file:
            if(self.sbatch):
                print(self.get_sbatch_config(), file=cmd_file)

            print(cmd_python, file=cmd_file)


        cmd_bash = '''
        chmod 755 {};
        ./{} &
        '''.format(cmd_path, cmd_path).replace("\n", "")


        if(self.sbatch):
            cmd_bash = '''
            chmod 755 {};
            sbatch {}
            '''.format(cmd_path, cmd_path).replace("\n", "")


        print("\n\n\n")
        print("Testing CMD : ", cmd_python)
        print("\n\n\n")


        p = subprocess.Popen(cmd_bash, shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

        process.append(p)

        p.communicate()

        exit_codes = [p.communicate() for p in process]



    def summaryRes(self, num_config=0):
        """
        This method make a summary for the benchmark
        """

        report = Report(self)
        report.make_report(num_config)
        report.toCSV(num_config)



    def launch_benchmark(self):
        """
        This method launch the benchmark process
        that has been configured by the csv file
        """

        os.makedirs(self.input_dir)

        for i in range(self.getNbTest()):

            self.getParam(i)

            os.makedirs(self.params['input_dir'])

            with open('{}/config_benchmark.txt'.format(self.params['input_dir']), "w+") as f:
                print(self.toString(), file=f)


            timeElapsed = time.time()
            self.launch_benchmark_data()
            print("Time elapsed configuring the data : ", time.time() - timeElapsed)

            timeElapsed = time.time()
            self.launch_benchmark_training()
            print("Time elapsed training : ", time.time() - timeElapsed)

            timeElapsed = time.time()
            self.launch_benchmark_testing()
            print("Time elapsed testing : ", time.time() - timeElapsed)

            self.summaryRes(i)




if __name__ == '__main__':

    file_name = 'res_brut.csv'
    sbatch = False

    try:
        file_name = sys.argv[1]
        sbatch = sys.argv[2] == "sbatch"
    except Exception as e:
        pass



    #Remove all argv (to remove error with the ArgumentParser in argument.py)
    sys.argv = sys.argv[:1]

    print("file_name : ", file_name)

    timeElapsed = time.time()

    benchmark = Benchmark(file_name, sbatch)
    print("Number of test : ", benchmark.getNbTest())
    benchmark.launch_benchmark()

    print("Time elapsed : ", time.time() - timeElapsed)
