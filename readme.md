# Benchmark: Prion Neurodegeneration Study

## This project takes a dataframe created from the Neuprint Janelia website and creates a graph where nodes organized by neuron types display feedforward neuron connection.

### Installation instructions 
1. Download the installer '.exe' from [Python Download](https://www.python.org/downloads/) to get the latest version of python. Type 'python' or 'python3' in command line to check if installation worked.

2. Pip should be automatically installed from step 1. Type 'pip help' in command line and see if you get error message. If error occurs, download script from [Pip Install](https://bootstrap.pypa.io/get-pip.py) and run 'python get-pip.py' or 'python3 get-pip.py' in command line

3. Install the following packages in command using following commands: pip install package
   1. numpy
   2. matplotlib
   3. networkx
   4. neuprint
   5. pandas

### Usage Example
The following examples show how the node shapes are created followed by an image of the finished product from Neurodegeneration part 2.py

    node_shapes = np.linspace(10, maxweight, 5)
    for node in graph.nodes():
        for i in node_shapes:
            if graph.nodes()[node]['size'] <= i:
                if i == node_shapes[0]:
                    graph.add_node(node, shape="o")
                    break
                elif i == node_shapes[1]:
                    graph.add_node(node, shape="D")
                    break
                elif i == node_shapes[2]:
                    graph.add_node(node, shape="8")
                    break
                elif i == node_shapes[3]:
                    graph.add_node(node, shape="s")
                    break
                elif i == node_shapes[4]:
                    graph.add_node(node, shape="p")
                    break

![](img_1.png)

### Development setup
This project has been documented by Sphinx, which is supported by Read the docs. In sphinx, I was able to make my APL documentation in HTML format. Sphinx-Test Reports shows test results inside Sphinx documentations and provides the following features: 

test-file: Documents all test cases from a junit-based xml file.

test-suite: Documents a specific test-suite and its test-cases.

test-case: Documents a single test-case from a given file and suite.

test-report: Creates a report from a test file, including tables and more for analysis.

test-results: Creates a simple table of test cases inside a given file.

test-env: Documents the used test-environment. Based on tox-envreport.

#### Usage of Test Suite

    .. test-suite:: My Test Suite
    :file: my_test_data.xml
    :suite: my_tested_suite
    :id: TESTSUITE_1
id: Unique id for the test file. If not given, generated from title.
file: file path to test file. If relative, the location of conf.py folder is taken as base-folder
suite: Name of the suite
status: A status as string
tags: a comma-separated list of strings
links: a comma-separated list of IDs to other document test_files/ needs-objects
collapse: if set to "True", meta data is collapsed.

Test suite automatically adds the following options: 

cases: amount of found cases in test file

passed: amount of passed test cases

skipped: amount of skipped test cases

failed: amount of failed test cases

errors: amount of test cases which have errors during test execution

time: needed time for running all test cases in suite 
### Meta
Anirejuoritse Egbe - lax18christian@gmail.com

Distributed under Applied Physics Laboratory license. See license for more information.