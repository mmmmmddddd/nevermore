
####### MLOps #######

train:
	python tools/train.py


####### DEVOps #######

dev:
	pip3 install -r requirements/develop.txt
	pre-commit install

build:
	python setup.py build

upload:
	python setup.py bdist_wheel upload -r hobot-local

clean:
	@rm -rf build dist *.egg-info

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
