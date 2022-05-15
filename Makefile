
####### MLOps #######

train:
	export PYTHONPATH=`pwd`; \
	python tools/train.py --config configs/baseline.py


####### DEVOps #######

dev:
	pip3 install -r requirements/develop.txt
	pre-commit install

build:
	python setup.py build

clean:
	@rm -rf build dist *.egg-info tmp_*

test:
	pytest -s tests

pep8:
	autopep8 nevermore --recursive -i

lint:
	pylint nevermore --reports=n

lintfull:
	pylint nevermore

install:
	python setup.py install

uninstall:
	python setup.py install --record install.log
	cat install.log | xargs rm -rf 
	@rm install.log
