======Prerequisites======
Platform: Linux / Windows

Language: Python 2.7 or greater

Packages :
	pandas (version 0.18.0)
	numpy  (version 1.10.4)
	scipy  (version 0.17.0)
	scikit-learn (version 0.17.1)
	matplotlib (version 1.3.0)

======Installation======
	Installing Python:
		Please follow the instructions given at following official documentation:
		https://docs.python.org/2/using/unix.html#getting-and-installing-the-latest-version-of-python

	Installing python packages:
		using pip:
			use "pip install <package_name>" to install above packages in python
			here <package_name> will be replaced by package's name 
		using easy_install:
			use "easy_install install <package_name>" to install above packages in python
			here <package_name> will be replaced by package's name

		pip and easy_install will work for 'pandas' and 'scikit-learn'. In order to setup numpy and scipy, 
		please follow the instructions given here:

		http://www.scipy.org/install.html

======Execution======
	The all_codes folder has following files:
		- finalSubmissionLogistic.py (our final submission on Kaggle site. It gives a csv file as output which has all the predicted values of 
									  survival on testing data)
		- logistic_cross.py          (code for applying cross_validation using logistic regression. It prints a mean of accuracies of all    
		  								iteration)
		- logistic_linear.py         (code for applying cross_validation using linear regression. It prints a mean of accuracies of all    
		  								iteration)
		- linear.py                  (submission code using linear regression. It gives a csv file as output which has all the predicted values 
									  of survival on testing data)
		- train.csv                  (training dataset)
		- test.csv                   (testing dataset)
	
	Execution order:
		Each of the above .py files will be executed as follows:
			python <file_name>

		The above files must be executed in following order:
		1) linear_cross.py   			(To observe cross_validation score on linear regression)
		2) logistic_cross.py 			(To observe cross_validation score on logistic regression)

		both of the above scripts will give cross_validation score using linear and logistic regression for comparison purposes
		
		3) linear.py         			(To perform linear regression and get prediction result on test data)
		4) finalSubmissionLogistic.py   (To perform logistic regression and get prediction result on test data)

		Both of the above files will put predictions in respective csv files in the format required by kaggle: 
			finalSubmissionLogistic.csv for logistic regression
			finalSubmissionLinear.csv   for linear regression



