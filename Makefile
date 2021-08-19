
dev:
	pip3 install -r requirements/develop.txt
	pre-commit install

build:
	python setup.py build

upload:
	python setup.py bdist_wheel upload -r hobot-local

clean:
	@rm -rf build dist src/*.egg-info

test:
	python /usr/bin/nosetests -s tests --verbosity=2 --rednose --nologcapture

pep8:
	autopep8 src/nevermore --recursive -i

lint:
	pylint src/nevermore --reports=n

lintfull:
	pylint src/nevermore

install:
	python setup.py install

uninstall:
	python setup.py install --record install.log
	cat install.log | xargs rm -rf 
	@rm install.log
